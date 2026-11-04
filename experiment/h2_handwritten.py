"""
H2：由 LLM（Claude）直接写出的上下文补全模型。没有梯度下降，没有读训练集原文，
词频、语法、搭配全部凭对英文和维基百科文体的了解手写。

结构（每个目标词开始时算一次 P(下一个词 | 前文)，之后每键只做前缀筛选）：

  第 0 层  前缀层    W·x 精确匹配（与 B0 相同的手写规则），由 ScoreModel 用字母序区间实现
  第 1 层  先验      b[w] = Zipf(w)·ln10，Zipf 值按层手写；没写到的词按词形从词根推，再不行按长度给默认值
  第 2 层  语法头    前一个 token 的类别 c → 候选词类别的偏好 T[c, class(w)]；a / an 再按首字母读音加减分
  第 3 层  搭配头    手写的 1～3 词上下文 → 下一个词的概率表，按"长的覆盖短的"逐层插值
  第 4 层  复制头    段落里已出现过的词按出现次数加概率（cache）
  第 5 层  归纳头    上一次出现"前一个词"之后跟的词加概率（A B … A → B；前两个词都对上时权重更大）

  P_语法(w)  = softmax(b + T[c_prev, class(w)] + 冠词修正)
  P_搭配     = 从最短的上下文到最长的依次：P ← (1-Σp)·P + 表里列出的 p
  P(w|前文) = (1-λc-λi)·P_搭配(w) + λc·P_复制(w) + λi·P_归纳(w)
"""

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from models import ScoreModel, load_shared_vocab

VERSION = "v2"
# v2：数字后面加月份（"14 September"）、in / on 后面加月份；形容词词根推复数时多扣分（firsts）
# v1：修正词形推导（不再从功能词推出 hi/wa/andes/inning 这类词）；副词透明（看副词前面那个词定语法）；
#     归纳头对功能词减弱；补一批"动词/名词 + 介词"固定搭配
LN10 = math.log(10)
WORD_RE = re.compile(r"^[a-z]+$")
YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")

# ═══════════════════════ 第 1 层：词频先验（Zipf 值 = log10 每十亿词出现次数，按维基百科估计） ═══════════════════════

ZIPF_TIERS = [
    (7.8, "the"),
    (7.45, "of and"),
    (7.35, "in"),
    (7.3, "to a"),
    (7.0, "was"),
    (6.85, "is for on as by with"),
    (6.8, "he that"),
    (6.7, "at from his it"),
    (6.6, "an were which"),
    (6.5, "are also this be first had after"),
    (6.4, "or has their its not but one been two"),
    (6.3, "they new who she her during have all time"),
    (6.2, "other there into more when later would only most him season year some three over"),
    (6.1, """years game between about while can out film up then world made both such several before under
     war where album them may many against team city could through since released second state until called
     series these part"""),
    (6.0, """well number song being became including early around people united four so used back end following
     however took best did included high five north south national american won show british along will each
     work music life york like same received band due members group known main based river began school home
     league church area old another last although age family final named lost government club six king led"""),
    (5.9, """large played left use public form without further power few order next much even still now just within
     died death father son wife day days times top line local single title history period century half off down
     if what how any those we i our us you place way name side point own"""),
    (5.8, """games player players character characters episode episodes story role book published version original
     release radio studio live recorded recording chart singles songs tour video television production directed
     director cast critics review reviews positive station route road highway county park building built town
     village island located west east army battle division forces force troops ship ships navy fleet captain
     general command sent attack killed men german french english japanese party election president minister
     member council political court law storm hurricane tropical winds damage"""),
    (5.7, """great small major total began continued remained returned moved joined served appeared considered
     described found gave given held brought produced created developed designed opened completed established
     founded formed replaced featured performed wrote written announced reported stated said added noted believed
     suggested decided allowed required caused followed reached passed increased involved provided seen taken
     become make take give get go come see find keep john william george henry charles james thomas london
     england france germany japan china india australia canada"""),
    (5.6, """young short long little important different similar original modern ancient popular famous successful
     royal military social religious natural human common special real true better greater higher lower former
     current present recent previous various available possible likely able certain particular central northern
     southern eastern western italian spanish russian chinese indian australian canadian irish scottish european
     african roman greek christian catholic man woman women child children mother daughter brother sister husband
     friend friends queen prince lord emperor bishop pope saint god leader leaders officer officers soldiers
     commander governor mayor judge coach manager writer author artist singer actor actress producer composer
     robert david michael paul peter richard edward smith ireland scotland wales europe america washington
     california texas michigan florida ohio virginia carolina zealand africa"""),
    (5.5, """month months week weeks night morning evening hour hours centuries decade decades seasons summer winter
     spring autumn january february march april june july august september october november december seven eight
     nine ten twenty hundred thousand million third fourth fifth country countries states cities areas region land
     water sea air light fire house room street bridge center centre field ground site position level rate control
     service company business money market industry trade development design plan project program system case fact
     art word words language names titles news paper magazine newspaper"""),
    (5.3, """match matches goal goals score scored points win wins loss victory defeat cup championship championships
     tournament round teams clubs football baseball basketball hockey rugby cricket olympic olympics medal record
     yards touchdown quarterback runs innings debut career professional college draft contract signed loan transfer
     appearances albums bands musical rock pop charts hot billboard track tracks guitar bass drums vocals lyrics
     sound records label concert performance performances videos versions peaked copies sold certified platinum
     gold uk films movie plot script scene scenes audience viewers rating ratings praise award awards nominated
     mixed negative aired broadcast premiered starring regiment battalion brigade corps unit units infantry
     cavalry artillery attacks attacked wounded captured enemy guns gun naval submarine aircraft operation
     operations campaign siege defense defence armor armour tons knots battleship battleships cruiser cruisers
     destroyer destroyers torpedo torpedoes launched commissioned decommissioned sunk sank admiral squadron
     flagship crew storms wind rainfall rain flooding depression landfall intensity cyclone mph km mi pressure
     intensified weakened dissipated extratropical typhoon formed roads highways interstate mile miles
     intersection junction trunkline traffic lane lanes designated alignment freeway bypass township department
     transportation species genus birds bird males male females female eggs food plants plant trees tree animals
     animal fish body length size color colour habitat prey tail wings fossil fossils specimens island islands
     mountain mountains lake coast valley ocean bay district province capital border population census residents
     households kingdom empire reign throne dynasty castle cathedral abbey temple gods goddess deities egyptian
     rebellion treaty planet mars earth surface orbit moon sun star stars tower floor floors architect style
     gothic roof walls stone brick gameplay console nintendo sega playstation developer developers levels
     enemies"""),
    (5.0, """paris berlin rome tokyo moscow vienna austria hungary italy spain russia mexico brazil egypt greece
     israel iraq iran korea vietnam pacific atlantic mediterranean chicago boston philadelphia pennsylvania
     illinois georgia alabama mississippi louisiana maryland massachusetts jersey indiana wisconsin minnesota
     missouri tennessee kentucky oregon colorado arizona nebraska kansas iowa oklahoma utah nevada mary elizabeth
     anne jones brown louis frederick albert arthur francis joseph harry jack frank walter alexander"""),
    (4.7, """ability access account action activities activity actual addition additional administration advance
     advanced advantage affair affairs agency agent agreement aid aim airport alliance allies alone amount analysis
     angle announcement annual answer appearance application approach approval approximately arm arms arrival
     article artists arts aspect aspects assault assembly assistance association attempt attempts attention
     authority authorities average background balance ball bank banks bar base basic basis battles beach bed
     beginning behavior behaviour belief bell birth blood board boat boats border box boy boys branch brief
     brothers budget buildings burial camp candidate capacity car cars care cases cause cells chain chairman
     challenge champion chance change changes chapter charge chief choice circle citizens civil claim claims class
     classes climate coat collection colony combat comedy commercial commission committee communication
     communities community comparison competition complex concept concern concerns conditions conference conflict
     congress connection construction contact content contest context contrast contribution conversion copy core
     corner cost costs count couple course courts cover creation crime crisis criticism crowd culture customers
     cut dance data date dead deal debate debt decision deep defeat degree delay demand depth deputy description
     desire detail details device direction directors discovery disease display distance doctor document door
     drama dream drive drug duty economy edge edition editor education effect effects effort efforts egg element
     elements emergency emphasis employees energy engine engineer engines entrance entry environment equipment era
     escape estate estimate event events evidence example examples exchange exhibition existence expansion
     experience experiment expert explanation exposure extension extent face facilities facility factor factors
     failure faith fall fame fans farm fashion feature features feet festival fiction fighter fighting figure
     figures finance finish flag flight flow focus folk foot forest forms fort fortune foundation frame freedom
     front fuel function fund funding funds future gain gallery garden gas gate generation girl girls glass grade
     grand grant grass grave grounds growth guard guest guide hair hall hand hands harbour harbor head headquarters
     health heart heat height help heritage hero hill hills hit hole honor honour hope horse horses hospital host
     hotel houses housing ice idea ideas identity image images impact importance income increase independence
     influence information injury inspiration institute institution interest interior interview introduction
     investigation iron issue issues item items job jobs journal journey jury justice key kind knowledge labour
     labor lack lady landing languages laws lead leadership leave legend lesson letter letters library license
     limit limits lines link list literature location loss lot love machine majority management map marriage mass
     master material materials matter meaning means measure media medical medicine meeting memory mention message
     metal method methods middle mind mine minute mission model models moment monument motion mouth move movement
     movements murder museum nation nations nature neck network note notes novel object objects observation
     occasion offer office officials oil opening opera opinion opponent opponents opportunity opposition option
     orchestra organization organisation origin origins outcome owner owners page pain painting pair palace
     parents parish parliament participation parties partner partners passage passengers past path patients
     pattern peace person personal personnel phase philosophy photograph physical picture piece pieces pilot pitch
     plane planning plans platform play plays poem poems poetry police policy politics pool port portion portrait
     possibility post potential practice presence press price prices pride priest prime principle prison
     prisoners prize problem problems process product products profile progress projects promotion property
     proposal protection protest publication purpose quality quarter question questions race races railway range
     rank rates reaction reader readers reality reason reasons reception recognition recovery reference reform
     regions relationship relations religion remains reports representative republic reputation request rescue
     research reserve residence resistance resolution resources respect response responsibility rest result results
     return revenue revolution rights ring rise risk rivers roles rule rules safety sale sales sample scale
     schedule scheme scholar scholars science screen search seat secretary section sector security selection sense
     sentence sequence servant session settlement sex shape share shop shore shot sight sign signal silver
     situation skin sky society soil soldier solution source sources space speaker speech speed spirit sport
     sports square staff stage standard standards start statement statue status step stock stop strategy strength
     stress structure struggle student students study subject success suit summit supply surgery survey symbol
     table tank target task tax teacher teaching technique technology temperature term terms territory test text
     theatre theater theme theory thing things thought threat ticket timber tradition train training transport
     travel treatment trial trip truck truth tube type types union unity universe university user value values
     variety vehicle vehicles victim victims view violence vision visit voice volume vote voters wall wave weapon
     weapons weather website weight wing winner wood workers works writers writing yard youth zone
     accept accepted achieved acquired acted adapted adopted agreed aimed applied appointed approached approved
     argued arranged arrested arrived asked assigned assumed attempted attended attracted awarded beat belonged born
     broke broken burned buried captured carried caught celebrated changed charged chose chosen claimed closed
     collected combined commented compared competed composed concluded conducted confirmed connected constructed
     contained contributed converted covered criticized criticised damaged dated declared declined defeated
     defended defined delivered demanded demolished denied departed depicted destroyed determined directed
     disappeared discovered discussed dismissed displayed dropped earned elected emerged employed encountered ended
     engaged enjoyed entered escaped estimated expanded expected explained expressed extended faced failed fell
     felt filled filmed finished fired fled flew forced fought gained granted grew headed heard helped hired hit
     hosted identified improved indicated influenced inspired installed intended introduced invaded invited issued
     kept knew landed lasted learned limited linked listed lived managed married marked meant met modified named
     needed observed obtained occupied occurred offered opposed ordered organized organised owned paid painted
     participated placed planned praised prepared presented preserved prevented printed proceeded promoted
     proposed protected proved published purchased raised ran realized recognized recognised recommended recovered
     reduced referred refused regarded rejected related relied remembered removed renamed repaired represented
     requested rescued resigned resulted retained retired revealed rose ruled sailed saw scheduled secured seemed
     selected settled shared showed shown sold sought spent spoke spread started stayed stood stopped struck
     studied succeeded suffered supported surrounded survived taught tested thought threatened told toured trained
     transferred translated travelled traveled treated tried turned understood used visited voted walked wanted
     warned watched went withdrew worked according having making taking playing working using becoming leading
     leaving coming going getting giving trying starting running moving setting showing serving remaining
     continuing operating producing writing attempting active afraid alive apparent attractive aware beautiful
     bright broad busy capable chief clear close cold comic complete considerable constant contemporary critical
     daily dangerous democratic difficult direct distinct dominant double dramatic dry effective electric
     electronic entire equal essential exact excellent existing expensive external extensive extra extreme fair
     federal final fine flat foreign formal free frequent fresh global golden grey guilty happy hard heavy historic
     historical huge immediate independent individual industrial initial intense internal joint junior largest
     latest legal limited literary male massive minimum moderate modest multiple native necessary notable nuclear
     numerous obvious official ordinary overall permanent poor powerful practical primary principal prior
     prominent proper quick quiet rapid rare ready regional regular relative remote responsible rich right rival
     rural safe scientific secondary secret senior separate severe significant simple slow soft solid specific
     strange strong subsequent substantial sudden suitable superior technical temporary terrible thick thin
     traditional typical unable unique unknown unusual upper urban useful usual valuable vast violent visible weak
     wide wild wooden worth wrong"""),
    (4.3, """assists batting bowl bowling coaches defensive offensive pitcher pitched pitching playoff playoffs
     semifinal stadium striker midfielder defender goalkeeper fixture premier relegated promoted relegation squad
     referee tackles rushing passing receiving interception interceptions nfl ncaa nba nhl mlb uefa fifa wicket
     wickets acoustic airplay ballad chorus remix verse songwriter soundtrack demo rapper hip hop rhythm melody piano
     keyboard symphony concerto screenplay cinematography grossed sequel trailer premiere dvd filming footage
     animation animated sitcom arcade graphics multiplayer ammunition dreadnought frigate frigates warship
     warships ironclad gunboat convoy convoys escort escorted bombardment raid raids sorties counterattack
     casualties surrender surrendered retreat retreated gusts inches advisories warnings evacuated flooded floods
     precipitation northbound southbound eastbound westbound concurrency interchange exit exits mushroom fungus
     fungi spores larvae shells shell crustacean lobster crab fins feathers plumage nest breeding mating predators
     diet insects asteroid comet galaxy telescope deity worship ritual priests temples pharaoh osiris"""),
]
DEFAULT_ZIPF = 3.4        # 没写到、也推不出来的词
LENGTH_SLOPE = 0.05       # 默认值：词每长一个字母（超过 6 个）减多少

# ═══════════════════════ 第 2 层：语法头 ═══════════════════════

CLOSED = {
    "DET": "the a an this these those",
    "POSS": "his her its their our my your",
    "QUANT": """some any no every each all both many much more most several few other another such same various
             numerous less least enough either neither""",
    "PREP": """of in on at by for with from into onto upon about above across after against along among around
            before behind below beneath beside besides between beyond despite down during except inside near off out
            outside over past per since through throughout till toward towards under underneath unlike until up via
            within without amid amongst like than as""",
    "TO": "to",
    "CONJ": "and or but nor yet so",
    "SUB": "although though because while when whereas if unless whether where once whilst since after before until",
    "PRONS": "he she it they we i you there",
    "PRONO": "him them us me himself herself itself themselves myself yourself ourselves it you her",
    "BEF": "is was are were am",
    "BEB": "be",
    "BEP": "been being",
    "HAVF": "has had",
    "HAVB": "have having",
    "DO": "do does did",
    "MODAL": "will would can could may might shall should must cannot",
    "REL": "which who whom whose that what how why",
    "ADV": """also not never only then still however later often already now even just very too again almost always
           ever soon thus therefore instead nearly mostly largely mainly highly well rather quite perhaps sometimes
           usually generally eventually finally initially originally previously recently subsequently currently
           formerly here further together away back forward else indeed especially particularly approximately""",
    "NUMW": """one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
            seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million
            billion dozen""",
}
CLOSED_EXTRA = {"may": ("NN",), "march": ("NN",), "like": ("VB",), "well": ("NN",), "that": ("DET",)}

ADJ_WORDS = """new old good great high low large small long short big little young early late major minor main
 important different similar own whole full free strong weak heavy light dark black white red green blue golden open
 public private local national international royal military political economic social cultural religious natural
 human common general special original modern ancient medieval classical popular famous successful favorable
 favourable positive negative critical commercial total final real true best better worse worst greater larger higher
 lower former current present future recent previous next last single double available possible likely unable able
 due further certain particular specific central northern southern eastern western british american english french
 german italian spanish russian chinese japanese indian australian canadian irish scottish welsh european african
 roman greek christian catholic jewish muslim first second third fourth fifth sixth seventh eighth ninth tenth top key
 hot cold wide deep fast slow hard easy simple serious difficult significant notable"""
VB_WORDS = """become include receive allow provide remain appear consider describe establish require contain continue
 believe seem mean refer prevent involve achieve avoid produce reach replace leave join meet bring keep give take make
 get go come see know find think tell ask feel let begin win lose build send hold write read buy sell pay die kill
 destroy enter create develop serve perform represent maintain improve protect"""
NV_WORDS = """play use work record release attack return start end change turn report support form set run lead fight
 visit control design plan move show call need help rule place name act award cover feature land launch love hope
 sign order rank score charge claim command cost count cut deal drive escape experience face fire focus head host
 issue lack mark match note offer pass point power press process promise question rain range rate rest result ride
 rise risk round sail share shift shot sound stand step stop strike study tour trade train travel trust view vote
 walk watch wish track base contract cause defeat demand display draw drop fall finish fly force guard hit increase
 interest list look matter mind object picture pull push rank record reply roll sort spread stay support switch
 talk test touch transfer vote win"""
VBD_WORDS = """became began came went saw gave found left led won lost wrote held met brought built fought sold told took
 made said paid ran rose fell grew knew drew flew threw chose spoke broke stood sent spent kept felt heard thought
 taught caught bought sought struck hit set put let laid lay sank sang swam done gone seen given taken written known
 shown grown drawn thrown chosen spoken broken stolen fallen risen driven born borne beaten forgotten hidden ridden
 sworn torn worn won shot fed bred led wound bound"""
LY_NOT_ADV = set("family italy july holy reply supply apply rely ally rally assembly anomaly monopoly early daily likely only weekly lonely ugly".split())
ING_NOT_VBG = set("king thing spring string ring wing morning evening ceiling wedding beijing everything nothing something anything sing bring sterling viking pudding".split())
ED_NOT_VBD = set("red bed need speed seed breed feed shed hundred sacred naked wicked kindred".split())

NEXT_CLASSES = ["DET", "POSS", "QUANT", "PREP", "TO", "CONJ", "SUB", "PRONS", "PRONO", "BEF", "BEB", "BEP", "HAVF",
                "HAVB", "DO", "MODAL", "REL", "ADV", "NUMW", "ADJ", "NN", "NNS", "VB", "VBD", "VBG", "VBZ"]

# 前一个 token 的类别 → 下一个词各类别的加分（自然对数；没写的为 0）
_DET_ROW = dict(ADJ=1.5, NN=1.5, NNS=1.0, VBG=0.3, VBD=0.3, NUMW=0.5, QUANT=0.3, ADV=-0.7, VB=-0.8, VBZ=-1.5,
                PREP=-3, TO=-3, CONJ=-3, SUB=-3, PRONS=-4, PRONO=-4, BEF=-4, BEB=-4, BEP=-4, HAVF=-4, HAVB=-4,
                DO=-4, MODAL=-4, REL=-3, DET=-4, POSS=-4)
_NN_ROW = dict(PREP=1.5, BEF=1.2, HAVF=0.8, MODAL=0.6, CONJ=1.0, VBD=0.9, VBZ=0.6, REL=0.8, NN=0.3, NNS=0.3, SUB=0.3,
               TO=0.5, ADV=0.3, DET=-2.0, POSS=-2.0, ADJ=-0.8, PRONS=-1.5, PRONO=-2, VB=-0.8, VBG=-0.2, NUMW=-0.3,
               BEB=-3, BEP=-2, HAVB=-1, QUANT=-1)
_BE_ROW = dict(VBD=2.2, ADJ=1.2, DET=1.0, ADV=0.8, VBG=0.6, PREP=0.6, NUMW=0.3, TO=0.3, QUANT=0.3, NN=-0.3, NNS=-0.5,
               BEF=-4, BEB=-4, BEP=-3, HAVF=-4, HAVB=-4, MODAL=-4, DO=-4, VB=-2, VBZ=-3, PRONS=-2.5, CONJ=-1.5)
_HAVE_ROW = dict(BEP=3.0, VBD=2.5, DET=1.2, POSS=0.6, QUANT=0.8, NUMW=0.6, ADV=1.0, TO=0.8, NNS=0.3, PREP=-0.3,
                 VBG=-1, VB=-1.5, VBZ=-2, BEF=-3, BEB=-3, HAVF=-2, HAVB=-2, MODAL=-4, CONJ=-1.5, PRONS=-2, PRONO=-0.5,
                 REL=-1)
_VB_ROW = dict(DET=1.5, POSS=1.0, PREP=0.8, PRONO=1.2, QUANT=0.6, NN=0.3, NNS=0.5, ADV=0.5, TO=0.5, ADJ=0.3, NUMW=0.5,
               VBG=-0.5, VBD=-1.5, BEF=-2, MODAL=-3, PRONS=-1.5, CONJ=-0.3, BEB=-2, BEP=-2, HAVF=-2)
TRANS = {
    "DET": _DET_ROW,
    "POSS": {**_DET_ROW, "NNS": 1.2},
    "QUANT": dict(NN=0.8, NNS=1.5, ADJ=1.0, PREP=0.8, DET=0.5, POSS=0.3, NUMW=0.3, QUANT=-0.5, VB=-1, VBZ=-1, BEF=-0.5,
                  PRONS=-2, PRONO=-1, MODAL=-1.5, CONJ=-1),
    "PREP": dict(DET=2.0, POSS=1.5, NUMW=0.8, QUANT=0.8, ADJ=0.5, NN=0.3, NNS=0.5, VBG=0.8, PRONO=1.0, REL=0.5,
                 PRONS=-1.5, BEF=-3, BEB=-2, BEP=-2, HAVF=-3, HAVB=-2, MODAL=-4, DO=-3, PREP=-1.5, CONJ=-3, SUB=-2,
                 TO=-2, VB=-1.5, VBD=-1, VBZ=-2, ADV=-0.5),
    "TO": dict(VB=1.8, DET=1.5, POSS=1.0, BEB=1.5, HAVB=0.8, NUMW=0.3, PRONO=0.5, QUANT=0.3, ADJ=-0.3, NNS=-0.3,
               VBD=-2, VBG=-0.5, VBZ=-2, BEF=-4, HAVF=-3, BEP=-3, MODAL=-4, PRONS=-1.5, CONJ=-3, PREP=-2, TO=-4,
               SUB=-2, REL=-1),
    "CONJ": dict(DET=1.2, POSS=0.6, ADJ=0.5, NN=0.3, NNS=0.4, VBD=0.6, VB=0.2, PRONS=0.8, ADV=0.4, QUANT=0.3, PREP=0.3,
                 NUMW=0.3, BEF=-0.5, HAVF=-0.3, MODAL=-0.5, CONJ=-3, SUB=-0.5, BEB=-2, BEP=-2),
    "SUB": dict(PRONS=1.8, DET=1.5, POSS=0.5, NN=0.3, VBG=0.3, ADV=0.3, QUANT=0.5, BEF=-0.5, VBD=-0.3, CONJ=-3,
                PREP=-1, MODAL=-2, BEB=-2, BEP=-2, HAVF=-1, TO=-2, VB=-1),
    "PRONS": dict(BEF=2.0, HAVF=1.8, HAVB=1.0, MODAL=1.8, VBD=2.0, VBZ=1.2, DO=1.0, ADV=1.2, NN=-2, NNS=-2, ADJ=-2,
                  DET=-3, POSS=-3, PREP=-1.5, PRONS=-4, PRONO=-3, CONJ=-1.5, TO=-2, REL=-2, VBG=-1, BEB=-3, BEP=-3,
                  NUMW=-2, QUANT=-1.5),
    "PRONO": dict(PREP=1.5, TO=1.2, CONJ=1.0, DET=0.5, ADV=0.5, POSS=0.3, VBG=-0.5, BEF=-0.5, MODAL=-1.5, VBD=-0.5,
                  NN=-1, ADJ=-0.8, PRONS=-1, VB=-0.5),
    "BEF": dict(VBD=1.8, VBG=1.2, ADJ=1.2, DET=1.2, ADV=1.2, NUMW=0.5, QUANT=0.5, PREP=0.6, POSS=0.3, TO=0.2, NNS=-0.2,
                BEF=-3, BEB=-3, BEP=-1, HAVF=-3, HAVB=-3, MODAL=-4, DO=-4, VB=-1.5, VBZ=-2.5, PRONS=-1.5, PRONO=-1.5,
                CONJ=-1, REL=-1.5, SUB=-1),
    "BEB": _BE_ROW,
    "BEP": {**_BE_ROW, "VBG": 0.8, "PREP": 0.8},
    "HAVF": _HAVE_ROW,
    "HAVB": _HAVE_ROW,
    "DO": dict(ADV=2.0, VB=1.5, PRONS=0.5, DET=0.3, NN=-1, VBD=-2, BEF=-3, MODAL=-3),
    "MODAL": dict(BEB=2.5, HAVB=1.8, VB=2.2, ADV=1.5, DO=-2, VBD=-3, VBG=-3, VBZ=-3, NN=-2.5, NNS=-2.5, ADJ=-2, DET=-2,
                  POSS=-2, PREP=-2.5, MODAL=-4, PRONS=-2, BEF=-4, BEP=-4, HAVF=-4, CONJ=-3, TO=-2, QUANT=-1.5,
                  NUMW=-2, REL=-3),
    "REL": dict(BEF=1.8, HAVF=1.5, MODAL=1.2, VBD=1.5, VBZ=0.8, PRONS=1.2, DET=1.2, POSS=0.5, ADV=0.8, ADJ=-0.5,
                PREP=-0.8, CONJ=-2, REL=-3, TO=-1.5, BEB=-2, BEP=-2),
    "ADV": dict(VBD=1.2, ADJ=0.8, VB=0.3, BEF=0.3, PREP=0.5, DET=0.5, TO=0.3, VBG=0.5, ADV=0.2, MODAL=-0.5, NN=-0.3,
                NNS=-0.3, PRONS=0.3, CONJ=-0.5, BEB=0.3),
    "NUMW": dict(NNS=2.0, NN=0.3, PREP=1.0, ADJ=0.5, CONJ=0.5, NUMW=0.3, DET=-2, POSS=-1.5, PRONS=-2, MODAL=-1, VB=-1,
                 VBZ=-1),
    "ADJ": dict(NN=1.5, NNS=1.5, ADJ=0.2, CONJ=0.5, PREP=0.5, TO=0.5, DET=-2.5, POSS=-2.5, BEF=-0.5, VBD=-0.5,
                PRONS=-2.5, PRONO=-1.5, MODAL=-1.5, VB=-1, VBZ=-1, VBG=-0.5, REL=-1, BEB=-2, BEP=-2, HAVF=-1,
                HAVB=-1.5, QUANT=-1),
    "NN": _NN_ROW,
    "NNS": {**_NN_ROW, "NN": 0.0, "NNS": 0.0, "VB": 0.2, "VBZ": -0.5, "HAVB": 0.8, "HAVF": 0.6, "VBG": 0.0},
    "VB": _VB_ROW,
    "VBZ": {**_VB_ROW, "TO": 0.8, "NNS": 0.3},
    "VBD": dict(PREP=1.5, DET=1.3, POSS=0.8, PRONO=0.8, ADV=0.8, TO=1.0, NUMW=0.5, QUANT=0.5, CONJ=0.3, ADJ=0.2,
                VBG=-0.3, VBD=-0.5, BEF=-1.5, MODAL=-3, PRONS=-1, VB=-1, BEB=-2, BEP=-1, HAVF=-1.5),
    "VBG": dict(DET=1.5, POSS=0.8, PREP=1.2, PRONO=0.8, NN=0.3, NNS=0.5, ADV=0.3, QUANT=0.5, TO=0.5, BEF=-1.5,
                MODAL=-3, PRONS=-1.5, VBD=-1, BEB=-2, HAVF=-1.5),
    # 标点与数字
    "START": dict(DET=1.5, PREP=1.2, PRONS=1.3, ADV=0.6, SUB=0.8, POSS=0.8, QUANT=0.3, NUMW=0.2, CONJ=-0.5, TO=-2,
                  BEF=-3, BEB=-3, BEP=-3, HAVF=-3, HAVB=-2, MODAL=-3, VBZ=-2, VB=-1.5, VBD=-0.8, VBG=-0.5, PRONO=-3,
                  REL=-1),
    "COMMA": dict(CONJ=1.5, REL=1.3, SUB=0.8, DET=0.8, PREP=0.8, PRONS=0.6, VBG=0.7, VBD=0.5, ADV=0.6, POSS=0.3,
                  NN=-0.3, NNS=-0.2, ADJ=-0.3, BEF=-1.5, HAVF=-1.5, MODAL=-2, TO=-1, VB=-1, BEB=-2, BEP=-2, PRONO=-1.5),
    "OPEN": dict(ADV=0.8, VBD=0.8, DET=0.6, NN=0.3, PREP=0.3, NUMW=0.3, CONJ=-1, BEF=-2, MODAL=-2, PRONS=-1, TO=-1),
    "CLOSE": dict(PREP=1.0, CONJ=1.0, BEF=1.0, VBD=0.8, REL=0.5, HAVF=0.5, DET=-0.5, NN=-0.5, ADJ=-0.5, POSS=-0.5),
    "QUOTE": dict(DET=0.3, NN=0.2),
    "COLON": dict(DET=1.0, PRONS=0.5, PREP=0.3, BEF=-1, MODAL=-1),
    "DASH": dict(ADJ=0.5, NN=0.8, VBD=0.8, VBG=0.5, NNS=0.3, DET=-2, POSS=-2, PREP=-1, BEF=-3, MODAL=-3, PRONS=-3,
                 CONJ=-2, REL=-2),
    "POSSM": dict(ADJ=1.2, NN=1.5, NNS=1.2, VBD=0.3, VBG=0.5, NUMW=0.5, QUANT=0.3, DET=-3, POSS=-3, PREP=-2, BEF=-1,
                  CONJ=-2, PRONS=-3, MODAL=-3, TO=-2),
    "NUM": dict(NNS=1.5, NN=0.5, PREP=0.8, CONJ=0.8, TO=0.6, ADJ=0.3, DET=-1, PRONS=-1, MODAL=-1),
    "YEAR": dict(PREP=0.5, CONJ=0.8, NN=0.5, REL=0.5, SUB=0.5, DET=0.3),
    "OTHER": {},
}
A_BEFORE_VOWEL = -3.0       # "a" 后面接元音开头的词
AN_BEFORE_CONSONANT = -4.0  # "an" 后面接辅音开头的词

# ═══════════════════════ 第 3 层：搭配表（上下文 → "词 概率 词 概率 …"，概率是"下一个是英文单词"时的条件概率） ═══════════════════════

NGRAMS = {
    # —— 标点、数字 ——
    ".": """the .17 in .06 he .055 it .04 this .028 however .028 a .025 she .028 they .022 after .018 on .014 as .014
          his .014 during .014 there .011 although .01 at .011 by .009 following .008 while .007 when .008 these .006
          according .006 for .007 with .006 its .006 their .006 many .005 some .005 one .005 from .005 despite .004
          both .004 since .004 an .005 two .004 several .004 later .004 because .004 we .003 i .003""",
    ",": """and .2 the .065 which .05 but .04 who .03 a .028 he .024 with .02 in .02 as .02 while .015 including .015
          where .012 although .01 it .01 his .01 she .008 they .008 its .006 when .008 after .008 was .006 to .01
          for .008 on .008 at .008 by .006 from .006 that .006 an .005 their .006 both .004 then .006 however .006
          or .008 so .004 being .004 one .004 two .004 such .004 most .004 making .004 only .004 not .004
          especially .004 though .004 whose .003 is .004 has .004 had .004 were .004 this .006 there .004""",
    "(": """also .04 the .04 born .03 now .02 a .02 known .015 or .01 later .01 which .01 including .01 and .006 in .01
          from .006 formerly .006 see .006 meaning .006 he .005 it .005 for .005 as .005 approximately .006 about .006
          an .004 on .004 at .004 with .004 by .004""",
    ")": """and .1 the .05 was .06 is .05 in .05 of .04 a .02 to .03 which .03 with .02 as .02 for .02 on .02 that .02
          at .015 from .015 by .015 he .01 it .01 were .01 are .01 has .01 had .01 or .01 but .01 who .006 when .005
          where .005 she .005 they .005""",
    '"': """the .05 a .03 i .025 and .02 was .02 in .02 to .02 of .02 is .01 as .015 by .01 it .01 we .01 you .01
          for .01 which .01 with .01 on .01 that .01 he .008 this .006 an .005 his .005""",
    "'s": """first .03 own .015 second .012 death .01 father .008 most .008 life .006 career .006 music .006 time .006
          album .006 family .006 debut .005 history .005 largest .005 main .004 mother .004 army .004 wife .004
          brother .004 work .004 name .004 son .004 team .004 performance .004 role .004 character .004 sister .004
          best .004 head .004 home .004 population .004 government .004 new .004 former .004 forces .004 body .004
          national .004 song .004 final .004 early .004 last .004""",
    "-": """year .03 based .02 time .02 old .02 known .015 term .01 class .01 american .01 length .01 called .01 made .01
          century .008 language .008 like .008 star .008 born .008 long .008 run .008 point .006 day .006 half .006
          sized .006 game .006 level .006 winning .006 style .006 scale .006 line .006 shaped .006 inch .006
          pounder .006 ton .006 up .01 out .006 off .006 the .01 member .004 law .004 speed .004 owned .004 led .004
          controlled .004 page .003 men .003""",
    ";": """the .1 he .04 it .03 a .03 this .02 they .02 in .02 she .02 however .02 and .02 his .015 there .01 these .01
          its .01 some .01 one .008 their .008 for .006 at .006 by .006 on .006 as .006 an .006""",
    ":": "the .1 a .03 in .02 he .015 it .015 this .01 they .01 one .01 his .01 i .01 we .006 an .006 their .006",
    "<num>": """years .04 and .03 to .025 in .03 the .02 of .025 people .015 km .03 mi .02 m .015 ft .015 million .03
          days .01 times .006 games .01 points .01 yards .015 goals .006 runs .006 per .01 metres .01 meters .006
          feet .01 miles .01 kilometres .006 inches .006 cm .006 kg .006 lb .006 hours .006 minutes .006 men .006
          copies .006 acres .004 episodes .004 on .01 for .01 at .008 with .01 by .01 as .006 from .008 was .008
          were .006 st .006 seasons .004 weeks .006 months .006 matches .004 appearances .006 tons .006 knots .004
          mph .006 percent .006 students .004 children .004""",
    "<year>": """and .06 the .04 to .04 when .02 he .02 season .03 election .01 in .03 by .02 for .015 with .015 at .01
          on .01 as .015 after .01 but .01 that .01 it .01 she .008 they .008 census .015 film .008 world .008 a .01
          until .008 where .006 which .008 was .01 were .006 had .006 his .006 olympics .006 summer .006
          atlantic .015 pacific .01 fifa .006 general .006 american .006 national .006 nfl .004 ncaa .004 there .005""",
    # —— 高频词 ——
    "the": """first .035 united .02 same .015 season .012 game .012 film .012 album .011 city .01 war .01 song .01
          end .009 world .009 following .008 second .008 new .008 team .008 band .007 series .007 episode .007 two .006
          original .006 british .006 church .006 river .006 north .006 state .006 national .006 main .006 most .005
          other .005 last .005 time .005 ship .005 storm .005 early .005 south .005 west .005 east .005 three .004
          final .004 royal .004 club .004 house .004 town .004 university .004 school .004 government .004 area .004
          island .004 battle .004 show .004 next .004 previous .004 character .004 species .004""",
    "of": """the .4 a .05 his .03 their .02 its .02 her .015 this .01 an .008 them .008 which .008 two .006 all .006
          these .005 those .004 england .004 new .004 one .004 being .004 three .003 some .003 several .003 time .003
          its .02 that .004 what .003 him .003 these .005 both .003 many .003 such .003""",
    "in": """the .35 a .05 his .02 which .02 their .015 an .01 order .01 its .01 this .01 addition .008 new .008
          response .005 particular .004 her .006 front .004 england .004 london .004 total .004 that .004 one .004
          some .004 two .004 part .004 early .004 late .004 several .003 many .003 both .003 contrast .003 fact .003
          place .003 practice .003 recent .003 time .003""",
    "to": """the .22 be .04 a .03 his .015 have .01 make .01 their .01 its .008 become .008 take .008 play .007 an .006
          them .006 her .006 help .005 him .005 return .004 use .004 reach .003 do .003 this .004 keep .003 prevent .003
          provide .003 support .003 attack .003 move .003 win .003 work .003 build .003 create .003 continue .003
          protect .003 replace .003 allow .003 see .003 find .003 give .003 get .003 go .003 form .003 join .003""",
    "and": """the .12 a .04 his .02 was .01 other .01 their .008 then .008 in .01 two .005 is .005 her .006 it .006
          had .006 has .004 an .005 also .006 its .005 later .005 they .004 he .008 she .004 to .006 that .005 on .005
          by .004 with .004 for .004 at .004 from .004 were .005 are .004 three .003 several .003 many .003 one .004
          after .003 several .003 became .003 eventually .003 more .003 most .003""",
    "a": """new .012 number .012 few .01 large .01 single .01 second .01 small .009 result .009 series .008 member .008
          major .007 total .007 short .006 long .006 similar .005 great .005 different .004 very .004 more .004
          high .004 local .004 group .004 private .004 game .004 year .004 time .003 two .003 film .004 song .003
          further .003 strong .003 good .004 young .004 small .009 significant .004 former .004 list .003 career .003
          record .004 period .003 role .003 team .003 song .003 week .003 month .003 day .003 man .003 woman .003""",
    "an": """american .06 english .03 important .02 additional .02 early .02 album .015 attack .015 average .015
          area .015 example .015 effort .015 independent .01 initial .01 international .01 old .01 open .01 eight .008
          estimated .01 episode .01 army .008 end .01 officer .008 event .008 interview .01 increase .008 unknown .008
          original .008 individual .006 island .006 election .006 annual .006 eastern .005 earlier .006 issue .006
          image .004 extended .004 award .006 order .004 unusual .004 extensive .004 alternative .004 entire .004""",
    "was": """a .07 the .06 not .035 also .035 released .03 born .02 named .015 built .012 an .012 one .01 used .01
          made .01 first .01 killed .008 later .008 in .02 to .02 originally .006 written .006 replaced .005 then .007
          still .005 able .006 given .005 eventually .004 completed .004 opened .004 appointed .005 elected .005
          promoted .004 awarded .004 sent .004 published .004 no .004 very .004 considered .004 only .005 his .006
          her .004 at .008 on .008 by .006 for .006 with .004 as .006 that .005 more .004 so .003 described .004
          designed .004 founded .004 forced .004 taken .004 found .004 placed .004 known .004""",
    "is": """a .09 the .07 also .04 not .03 an .015 one .015 located .012 used .01 known .008 described .005 still .005
          currently .005 considered .005 found .006 now .006 made .004 only .005 in .01 to .01 that .005 very .004
          often .004 based .004 part .004 believed .004 home .004 thought .004 called .004 named .003 composed .003
          set .003 played .003 more .003 most .003 about .003""",
    "for": """the .25 a .05 his .03 their .02 its .015 best .01 her .008 this .008 an .007 example .015 more .006
          two .005 which .006 him .004 use .003 all .004 being .004 over .004 them .004 three .003 several .003
          many .003 each .003 one .003 some .003 those .003 what .003 both .003 her .008 other .003 years .003""",
    "on": """the .4 a .04 his .02 their .01 its .01 her .006 this .006 which .008 an .005 may .004 television .003
          both .003 board .003 his .02 one .003 two .003 behalf .004 tour .003 top .003 their .01 each .003""",
    "as": """a .2 the .15 well .06 an .03 part .025 his .02 it .01 he .01 one .01 such .005 of .01 to .008 being .006
          early .006 many .006 much .006 long .006 soon .006 far .004 they .006 she .005 their .008 its .006 her .006
          president .003 head .003 captain .003 if .004 though .004 we .003 i .003""",
    "by": """the .35 a .04 his .03 their .01 its .01 an .008 this .006 which .006 her .006 two .006 british .004
          using .005 several .004 many .004 other .004 one .004 both .004 critics .006 american .003 german .003
          japanese .002 french .003 three .003 some .003 then .004 storm .002 local .003 being .003 far .004""",
    "with": """the .3 a .1 his .04 their .02 an .015 her .015 its .01 two .01 other .008 him .006 them .006 one .006
          more .006 only .006 several .005 many .005 some .005 three .004 each .004 no .004 four .003 most .003
          both .003 this .006 an .015 her .015 respect .003 regard .003 which .004""",
    "he": """was .25 had .1 is .03 also .04 would .03 has .02 did .015 could .01 made .01 played .01 became .01
          returned .008 took .007 then .008 said .008 began .008 wrote .007 died .007 won .006 left .006 received .005
          joined .005 married .005 later .006 never .006 continued .005 moved .005 scored .005 described .004
          stated .004 served .004 appeared .004 will .004 does .003 felt .003 decided .004 announced .003 spent .004
          led .003 used .003 met .003 found .003 gave .003 held .003 helped .003 believed .003 still .003""",
    "she": """was .25 had .1 is .03 also .03 would .03 has .02 did .01 could .01 then .008 later .006 became .008
          returned .006 married .005 began .006 made .006 received .005 left .005 died .005 sailed .004 arrived .004
          remained .004 continued .004 took .004 joined .004 served .004 wrote .004 appeared .004 said .004""",
    "it": """was .22 is .15 has .04 had .03 would .03 also .02 will .01 became .01 can .01 could .01 did .008 may .008
          received .008 reached .008 to .01 in .01 a .01 as .01 being .006 does .005 remains .004 the .006 seems .004
          took .004 made .004 debuted .004 peaked .004 features .003 includes .003 still .004 later .004 not .004
          for .004 on .004 with .004 at .004 from .004 into .003 should .003 must .003""",
    "they": """were .2 had .1 are .06 would .04 have .04 also .03 could .02 did .015 then .01 can .01 will .01 won .006
          found .006 began .006 played .006 returned .005 made .005 moved .005 lost .005 finished .005 still .004
          became .004 continued .004 decided .004 used .004 met .004 left .004 attacked .003 took .004""",
    "which": """was .15 is .08 the .06 had .04 he .03 were .03 has .03 it .02 they .015 are .02 a .01 she .008
          would .01 includes .006 included .006 became .006 can .006 also .006 could .006 in .01 led .006 made .004
          allowed .004 means .004 became .006 contains .004 features .004 took .004 resulted .004 caused .004
          remained .004 began .004 consists .004 then .004 still .004 will .004 may .004""",
    "who": """was .15 had .08 is .04 were .04 has .03 would .03 also .02 played .01 became .01 later .01 died .01
          are .01 have .01 then .006 served .006 could .006 won .005 did .005 wrote .005 lived .005 worked .005
          took .005 made .005 married .004 led .004 received .004 described .004 appeared .004 joined .004""",
    "that": """the .25 he .06 it .04 they .03 she .03 was .04 a .03 is .03 his .02 there .015 were .015 had .015
          this .01 would .008 in .01 time .008 year .008 their .008 could .005 are .01 its .005 an .005 if .004
          many .004 these .004 we .004 i .004 such .004 some .004 one .004 has .006 have .006 will .004 had .015
          season .004 period .003 same .003 night .003 day .003""",
    "at": """the .5 least .04 a .03 his .02 first .01 home .01 all .008 that .008 their .008 which .006 her .005 its .006
          this .006 an .004 one .004 times .004 once .003 night .003 number .004 two .003 sea .003 age .003 end .003
          different .003 various .003 several .003 school .003 university .003""",
    "from": """the .4 a .03 his .02 their .01 its .01 her .006 which .006 an .005 one .006 two .005 this .006 other .004
          new .004 being .004 both .004 all .004 them .004 him .003 several .003 three .003 each .003 there .004
          what .003 many .003 these .003 those .003""",
    "his": """own .03 first .025 father .02 wife .015 death .015 brother .012 career .012 life .012 mother .01 time .01
          son .01 family .01 return .006 work .008 second .006 home .006 position .005 name .005 men .006 early .006
          role .006 army .004 body .004 team .005 final .004 last .004 debut .005 successor .004 reign .004 head .003
          efforts .003 friend .004 uncle .003 daughter .005 sister .005 parents .004 brothers .004 father .02
          professional .004 previous .003 new .004 best .003 eldest .003 younger .003 older .003 personal .003
          political .003 military .003 return .006 arrival .003 command .003 forces .003 troops .003 ship .003""",
    "her": """first .02 own .02 husband .02 mother .015 father .012 life .01 career .012 debut .006 to .02 and .015
          death .006 sister .006 role .006 son .006 daughter .005 family .006 second .005 album .006 music .004
          work .005 voice .004 time .004 name .004 home .004 in .01 as .01 the .01 crew .004 maiden .004 guns .003
          sister .006 career .012 performance .004 brother .004 final .003 last .003""",
    "its": """own .03 first .03 name .015 original .01 way .006 debut .008 population .006 second .006 current .006
          use .006 final .006 highest .004 peak .004 release .006 members .004 history .006 development .004
          early .005 main .005 largest .004 southern .004 northern .004 eastern .004 western .004 location .004
          members .004 design .004 new .004 time .003 initial .004 entire .004 maximum .004 course .004 position .004
          size .004 construction .003 predecessor .004""",
    "their": """own .04 first .04 second .015 home .012 way .008 album .008 debut .008 lives .006 last .006
          families .006 final .006 children .006 position .005 positions .005 best .005 team .005 music .005 work .005
          new .005 next .005 attack .004 relationship .004 third .004 previous .004 time .004 home .012 fans .004
          return .004 members .004 parents .004 forces .004 troops .004 original .004 career .004 song .003""",
    "this": """time .04 is .05 was .05 period .02 album .01 song .01 point .01 year .015 season .01 film .008 game .008
          species .008 role .005 area .006 way .006 new .005 work .005 case .006 process .004 event .004 version .004
          type .004 name .004 has .006 led .008 had .005 would .005 can .004 one .004 means .004 includes .003
          campaign .004 incident .004 decision .004 attack .004 battle .004 storm .004 is .05 tour .004 book .004
          episode .004 character .004 series .004 group .004 building .004 church .004 day .004 week .004 month .004""",
    "be": """a .07 the .06 used .03 made .02 an .015 released .015 able .015 found .012 seen .012 played .01 more .008
          built .008 killed .006 taken .006 given .006 considered .006 in .01 very .006 replaced .005 called .005
          included .005 added .004 shown .004 named .004 held .004 closed .004 completed .004 allowed .004 one .005
          known .004 placed .004 sent .004 required .004 published .004 based .004 done .004 converted .004""",
    "been": """a .06 the .05 used .02 made .015 released .01 described .01 in .02 an .01 found .008 called .008
          able .008 built .006 involved .006 killed .006 established .005 identified .005 recorded .005 named .005
          suggested .004 reported .004 proposed .004 given .004 at .006 on .005 very .004 more .004 known .004
          nominated .004 awarded .004 replaced .004 seen .004 performed .004 compared .004 praised .004 criticized .004
          one .004 working .004 playing .004 taken .004 sold .004 the .05 completed .004 damaged .004 destroyed .004""",
    "had": """been .15 a .05 the .03 to .03 not .02 already .015 no .015 also .015 his .012 become .012 an .008
          previously .008 never .008 only .006 made .006 more .006 their .006 two .006 left .006 just .004 taken .004
          won .004 died .004 lost .004 begun .004 received .004 returned .004 played .004 given .004 little .004
          great .003 her .004 its .003 many .003 some .003 one .003 three .003 been .15 written .003 reached .003
          moved .003 served .003 gone .003 come .003 seen .003""",
    "has": """been .25 a .05 the .03 also .03 not .015 become .012 since .012 an .008 two .006 had .006 no .006 more .006
          received .006 its .006 his .005 made .005 appeared .004 won .004 said .004 never .004 only .004
          described .004 continued .003 released .003 one .004 three .003 led .003 remained .003 performed .003
          stated .003 written .003 recorded .003 sold .003 played .003 worked .003 taken .003 become .012""",
    "have": """been .25 a .03 the .03 to .03 also .015 not .012 had .008 become .008 no .008 an .006 made .006 their .006
          more .006 two .004 since .004 found .004 suggested .004 used .004 never .004 already .004 any .003 some .004
          many .003 led .003 described .003 come .003 seen .003 noted .003 argued .003 occurred .003 taken .003""",
    "were": """also .04 not .04 the .03 killed .03 made .015 a .015 used .015 built .012 released .01 sent .008 able .008
          later .01 in .02 found .01 taken .006 to .015 originally .005 given .006 still .006 then .006 eventually .004
          destroyed .005 added .005 replaced .005 forced .006 captured .005 published .004 no .006 only .006 all .006
          two .006 first .006 at .008 on .008 by .006 more .005 very .004 being .004 held .004 wounded .005
          recorded .004 placed .004 named .004 lost .004 removed .004 issued .004 assigned .004 unable .004 several .004
          many .004 present .004 three .004 an .004 his .004 their .004 some .004""",
    "are": """the .04 also .04 not .03 a .03 found .02 used .015 known .01 in .02 often .008 more .008 no .008 still .006
          many .006 two .006 made .006 now .006 able .006 considered .005 very .004 present .004 described .004
          generally .004 usually .004 located .004 all .005 only .005 being .004 currently .004 three .004 an .004
          most .004 typically .004 commonly .004 called .004 believed .004 thought .004 some .004 several .004""",
    "also": """a .05 the .04 been .03 known .02 used .015 released .015 appeared .015 made .01 had .01 received .01
          played .01 an .008 one .008 in .02 be .015 included .01 includes .006 found .008 became .008 won .008
          featured .006 noted .006 served .006 built .006 took .006 wrote .006 provided .005 has .008 considered .005
          began .005 worked .005 performed .005 reached .005 written .004 created .004 described .004 praised .004
          said .004 stated .004 very .004 his .006 two .005 several .004 her .004 on .006 at .006 to .006 for .006
          with .005 by .005 as .006 that .005 some .004 more .004 announced .004 suggested .004 introduced .004""",
    "first": """time .05 season .03 game .025 two .02 world .015 album .015 single .015 film .012 half .012 episode .01
          three .01 year .01 major .01 day .01 round .008 week .008 team .008 class .008 match .006 appearance .008
          one .006 series .006 used .008 published .008 released .006 to .01 in .02 of .01 and .01 as .008 with .005
          edition .005 book .005 person .005 president .004 official .004 full .005 woman .004 ever .004 public .004
          known .004 recorded .004 issue .004 quarter .005 goal .004 battle .004 national .004 lieutenant .004
          lady .003 four .004 five .004 part .004 performed .004 professional .003 league .003 international .003""",
    "one": """of .3 the .02 day .01 hundred .01 more .01 year .01 point .008 in .02 to .01 and .01 game .008 another .008
          other .008 or .006 that .006 which .005 was .01 is .008 who .004 with .005 week .005 season .006 night .004
          side .004 man .005 for .006 on .005 by .005 at .005 as .005 time .006 thousand .004 million .004""",
    "two": """years .06 of .04 other .02 days .02 weeks .015 months .015 children .012 games .012 hundred .01 more .008
          sons .008 men .008 different .008 and .02 to .01 in .01 teams .006 ships .006 hours .006 points .006 or .008
          were .006 new .006 species .006 albums .005 seasons .006 goals .005 brothers .004 daughters .004 players .005
          main .004 groups .004 decades .004 versions .004 songs .004 battalions .003 divisions .003 thousand .004
          million .004 separate .004 companies .003 regiments .003 sides .003 parts .004 types .004""",
    "three": """years .05 of .04 days .015 weeks .01 months .01 games .01 children .008 other .008 more .006 hundred .006
          men .006 people .006 ships .005 points .005 teams .005 albums .004 songs .004 seasons .005 sons .004
          types .004 main .004 different .004 consecutive .004 times .006 and .02 to .01 in .01 or .006 times .006
          goals .004 decades .004 divisions .003 battalions .003 thousand .004 million .004 companies .003""",
    "new": """york .2 zealand .05 england .02 south .015 jersey .015 orleans .012 album .01 world .008 guinea .006
          team .006 hampshire .006 mexico .008 version .006 government .006 species .006 church .005 building .005
          song .004 material .006 series .004 season .004 music .004 line .004 and .006 one .004 head .004 king .004
          city .004 owners .003 road .003 deal .003 era .003 president .003 contract .003 kingdom .003 testament .004""",
    "united": "states .6 kingdom .2 nations .08 artists .01 methodist .004 arab .004 airlines .004 front .003",
    "states": """and .03 army .03 navy .03 in .03 government .01 to .02 house .005 senate .006 that .008 census .006
          military .006 department .005 district .005 court .005 where .004 marine .004 air .005 congress .005
          naval .004 from .006 for .006 with .005 by .005 on .005 as .005 which .004 was .004 is .004""",
    "world": """war .45 cup .05 series .03 championship .02 championships .015 and .01 heritage .006 tour .006
          record .006 health .004 trade .004 wrestling .006 music .003 in .006 of .006 to .004 as .004 premiere .004
          tag .003 title .004 order .003 bank .003""",
    "war": """ii .25 i .1 and .02 in .03 of .015 was .01 he .006 between .006 against .006 with .006 to .008 began .005
          broke .004 ended .004 memorial .004 had .004 as .005 on .005 for .005 that .004 when .004 which .004 the .006
          office .004 effort .004 department .004 years .004 crimes .003 period .003""",
    "such": "as .7 a .06 an .02 that .01 things .004 was .004 the .006 large .003 high .003 good .003 great .003",
    "well": """as .4 known .1 received .05 in .01 to .01 and .01 documented .006 established .006 regarded .004 after .004
          before .004 into .004 over .004 be .003 liked .003 suited .003 beyond .004 enough .003 the .01 for .004""",
    "according": "to .95",
    "due": "to .9 in .01",
    "order": "to .45 of .1 in .05 the .02 for .02 was .02 that .02 and .01 from .01",
    "able": "to .9 and .01",
    "part": "of .7 in .05 time .02 the .01 and .01 as .01 one .01 two .01 three .01",
    "number": "of .55 one .12 two .06 three .02 five .01 four .01 six .006 seven .006 eight .006 nine .004 ten .004",
    "end": "of .6 the .02 in .01 and .01 up .02 with .01 to .01 zone .005 result .003 credits .003 date .003",
    "rest": "of .8",
    "most": """of .2 important .04 popular .03 common .02 successful .02 notable .02 famous .015 likely .015 recent .01
          significant .01 other .01 part .01 widely .008 commonly .008 powerful .008 prominent .008 valuable .006
          influential .006 frequently .006 difficult .006 expensive .006 people .006 players .004 critics .004
          notably .01 recently .008 in .01 successful .02 well .004 often .006 famous .015 severe .004 intense .004""",
    "more": """than .3 of .02 popular .01 important .01 likely .01 successful .008 money .006 people .006 time .006
          difficult .006 and .01 in .01 to .01 commonly .006 recently .006 information .004 complex .004 common .006
          powerful .004 recent .004 widely .004 closely .004 accurate .004 detailed .004 serious .004 effective .004
          traditional .004 recently .006 frequently .004 often .004 important .01 or .006 the .01 on .004 money .006""",
    "than": """the .1 a .08 any .02 one .02 two .02 in .03 that .02 his .01 its .01 their .01 ever .006 expected .006
          those .008 he .008 it .006 they .006 half .006 previously .004 usual .004 what .004 three .006 four .004
          five .004 ten .004 twenty .003 anticipated .003 originally .004 those .008 others .004""",
    "after": """the .35 a .06 his .06 being .03 their .02 her .02 an .01 he .02 which .02 two .02 its .02 she .008
          they .008 it .01 one .01 three .008 world .006 only .006 returning .006 this .01 that .01 several .006
          some .006 leaving .006 receiving .004 losing .004 completing .004 retiring .004 graduating .004 his .06
          years .004 four .004 five .004 six .003 many .003 seeing .003 hearing .003 learning .003 initially .003""",
    "during": """the .55 a .05 his .08 their .03 world .04 this .02 her .015 its .015 which .01 an .006 that .01
          these .005 one .004 its .015 production .004 filming .004 recording .004 both .004""",
    "before": """the .25 a .04 his .04 being .04 he .03 they .02 she .02 it .02 their .015 returning .01 its .01 her .01
          moving .006 an .006 that .01 this .01 finally .006 eventually .005 dying .003 retiring .004 leaving .005
          joining .004 becoming .004 making .004 taking .003 reaching .004 entering .004 any .003""",
    "between": """the .4 two .03 a .02 his .01 them .01 their .01 british .004 these .004 him .004 her .004 its .004 those .003
          rival .003 three .003 different .003 local .003 several .003 an .004""",
    "under": """the .45 a .04 his .05 their .02 construction .02 its .02 which .01 contract .008 pressure .006 fire .006
          control .006 her .006 command .01 an .005 new .004 this .004 british .004 several .003 consideration .003""",
    "against": """the .5 a .04 his .02 their .02 him .02 them .02 her .01 its .01 germany .006 other .005 british .006
          an .005 japan .004 france .004 england .004 any .003 all .003 two .003 local .003 rival .003""",
    "through": """the .5 a .04 his .02 their .02 its .01 which .01 an .005 her .006 several .005 two .004 this .006 what .003
          an .005 to .01 many .003 its .01 central .003 northern .003 southern .003 eastern .003 western .003""",
    "since": """the .25 it .04 its .03 then .06 he .03 they .02 she .015 a .02 there .01 their .01 his .015 been .006
          this .008 that .01 become .006 being .006 its .03 at .004 early .004 late .004 ancient .003""",
    "until": """the .4 his .06 he .04 a .03 it .03 they .02 she .02 her .02 their .01 its .01 after .008 late .006
          early .006 being .004 then .005 that .006 this .006 at .004 its .01 mid .004 now .004 recently .004""",
    "into": """the .45 a .05 his .02 their .02 an .015 its .015 two .01 her .01 one .006 service .006 account .006 law .004
          force .004 being .004 existence .004 production .004 orbit .004 space .004 office .004 three .004 several .003
          what .003 this .006 them .004 town .003 battle .003 action .003 play .003 power .003""",
    "about": """the .25 a .06 his .04 her .02 how .02 their .02 what .015 it .015 two .01 three .006 an .006 being .006
          this .01 half .008 him .006 them .006 one .006 five .004 four .004 ten .004 twenty .003 them .006 her .02""",
    "while": """the .15 he .08 she .04 a .03 they .03 his .02 it .02 in .02 others .006 also .006 on .01 playing .005
          working .005 there .005 still .006 at .01 serving .004 being .004 others .006 its .004 their .004 we .003""",
    "when": """the .25 he .12 she .05 they .05 it .05 a .04 his .02 their .01 there .01 an .006 asked .006 this .006 we .005
          i .005 you .004 its .004 her .004 one .004 two .003 compared .003 their .01 told .003 combined .003""",
    "where": """he .12 the .15 it .06 they .06 she .05 a .04 there .015 his .012 its .005 an .005 we .005 you .004 many .004
          several .003 their .006 her .004 one .004 two .003 this .004 these .003""",
    "although": """the .2 he .08 it .06 she .03 they .03 there .02 this .02 a .03 some .02 his .01 its .01 many .01 not .006
          no .006 most .005 both .005 these .005 its .01 an .006 one .004 two .003 their .006""",
    "however": """the .25 he .08 it .06 a .03 in .03 she .03 they .03 his .02 this .02 there .01 after .01 when .01 by .01
          its .006 as .008 on .006 due .006 following .005 since .006 many .005 some .005 both .005 at .006 their .006
          an .006 these .004 one .004 two .003 despite .004 during .004 with .004 for .004 we .003""",
    "because": """of .3 the .1 he .06 it .05 they .03 she .03 a .02 his .02 there .01 its .006 their .006 this .006 we .004
          i .004 many .003 some .003 most .003 no .003 her .004""",
    "if": """the .15 he .06 it .06 they .04 a .04 she .03 you .03 there .02 not .02 any .01 necessary .006 only .005
          we .005 i .005 this .005 their .004 his .004 one .004 any .01 an .004 no .004 so .004 possible .004""",
    "then": """the .1 a .04 he .03 moved .03 returned .02 became .02 went .015 used .01 began .01 known .01 released .01
          there .006 took .008 joined .006 turned .008 made .006 headed .006 proceeded .006 left .006 served .006
          won .006 played .006 spent .005 sent .005 attacked .004 continued .004 she .006 they .006 it .006 his .006""",
    "not": """be .05 a .04 the .04 only .05 have .02 been .02 to .02 in .02 as .01 yet .01 released .01 used .01 able .008
          until .008 allowed .006 received .006 known .006 included .006 make .006 want .005 seen .005 even .005
          always .005 necessarily .005 being .005 an .005 all .005 much .005 take .004 appear .004 made .004
          considered .004 found .004 return .004 well .004 completed .004 clear .004 so .004 for .006 on .004 at .004
          very .004 likely .004 known .006 recognized .003 affected .003 present .003 sure .003""",
    "would": """be .15 have .06 not .04 later .04 become .04 go .02 never .02 also .02 eventually .015 make .015 take .015
          continue .01 remain .008 return .008 play .008 win .006 include .006 allow .006 lead .006 like .006
          serve .005 appear .005 only .005 then .005 still .004 soon .004 use .004 receive .004 give .004 help .004
          need .004 come .004 get .004 see .004 find .004 replace .004 require .004 result .004 eventually .015""",
    "will": """be .25 not .03 have .02 also .02 continue .01 take .01 make .01 become .008 never .006 go .006 play .006
          return .006 always .005 only .005 still .004 remain .004 help .004 get .004 see .004 come .004 need .004""",
    "can": """be .3 also .02 not .015 only .015 still .01 make .01 cause .01 reach .008 take .008 do .006 have .008 see .006
          occur .006 produce .005 lead .005 grow .005 help .004 use .005 often .004 easily .004 even .004 affect .004
          result .004 vary .004 range .004 change .004 form .004 become .004 move .004 travel .004 survive .003""",
    "could": """be .2 not .08 have .05 only .015 also .015 see .01 make .01 take .008 no .006 do .008 get .006 never .006
          go .006 find .006 carry .006 reach .005 easily .004 use .004 become .004 still .004 even .004 help .004
          provide .004 return .004 hear .003 produce .003 lead .003 cause .003 support .003""",
    "may": """be .2 have .08 not .03 also .03 occur .01 include .008 refer .008 cause .006 lead .005 result .006 help .005
          affect .004 contain .004 only .005 never .004 still .004 well .004 the .02 he .006 it .006 and .01 in .01
          to .006 of .006 that .006 when .004 after .004 before .004 his .004""",
    "should": "be .25 not .05 have .03 also .01 never .006 only .006 take .006 make .006 go .006 receive .004 remain .004",
    "must": "be .25 have .06 not .03 also .01 then .006 first .006 take .006 make .006 find .006 use .006 defeat .004",
    "did": """not .55 so .02 the .02 he .01 well .01 it .01 they .006 this .006 she .006 a .01 more .004 his .004 her .004
          you .004 i .004 we .004 nothing .004 manage .003""",
    "there": """was .25 is .12 are .1 were .1 have .03 had .03 has .02 would .02 for .02 in .02 and .01 as .01 to .01
          will .01 until .006 being .006 he .006 they .005 it .005 remains .004 seems .004 she .004 may .004 might .004
          could .004 he .006 the .006 from .006 on .006 at .006 by .004 with .004 again .004 since .004""",
    "all": """of .15 the .12 three .02 four .02 his .03 their .03 other .02 time .015 its .01 but .01 these .01 that .01
          over .01 her .006 those .006 five .006 two .006 around .006 parts .005 aspects .004 sides .004 seven .004
          six .004 eight .004 star .006 members .005 directions .003 sorts .003 kinds .003 types .003 matches .003
          games .003 songs .003 episodes .003 known .003 nine .003 ten .003""",
    "both": """the .15 of .1 a .04 in .03 his .03 sides .02 teams .01 their .02 men .006 critics .006 countries .006
          british .005 from .008 on .008 as .008 to .008 at .006 by .006 for .006 with .006 its .006 her .006 an .006
          new .004 local .004 commercially .004 critically .004 players .003 ships .003 houses .003""",
    "some": """of .25 time .02 other .02 people .02 critics .015 sources .01 species .01 areas .01 years .01 cases .01
          point .01 way .006 form .006 members .008 historians .008 scholars .006 parts .006 two .006 three .006
          extent .006 kind .005 fans .005 reviewers .005 time .02 degree .005 years .01 others .006 instances .004
          countries .004 elements .004 aspects .004 changes .004 success .004 places .004 new .004 small .004""",
    "many": """of .2 other .04 people .03 years .02 more .015 times .01 different .01 critics .01 species .01 fans .006
          areas .006 were .006 others .01 as .01 new .006 ways .006 such .006 local .005 british .004 cases .004
          countries .004 parts .004 historians .004 reviewers .004 songs .004 members .004 players .004""",
    "several": """other .05 times .03 years .03 of .04 days .02 months .02 weeks .02 hours .01 different .01 new .01
          people .01 songs .008 species .008 ships .006 members .006 small .006 more .006 minor .005 major .005
          large .005 british .004 episodes .004 games .004 occasions .006 countries .004 others .004 books .003
          films .003 times .03 hundred .004 thousand .004 sources .004 critics .004 local .004""",
    "each": """other .15 of .1 team .03 year .03 game .02 episode .015 season .015 side .01 player .01 week .01 day .01
          time .01 song .008 member .008 case .006 with .01 in .008 had .008 was .008 has .008 one .006 country .004
          state .004 level .004 race .004 match .004 round .004 group .004 character .004""",
    "other": """than .06 hand .03 countries .02 members .02 species .02 parts .015 two .01 cities .01 players .01 states .01
          side .01 people .01 teams .01 ships .01 areas .01 things .006 forms .006 words .006 sources .005 times .005
          groups .006 characters .006 games .006 songs .006 artists .006 units .005 officers .005 major .006 works .006
          events .005 three .004 european .004 local .004 hands .004 similar .004 bands .004 nations .004 hand .03""",
    "another": """two .03 one .02 player .01 attack .01 game .01 three .01 man .006 way .006 version .006 year .006 season .006
          character .006 member .006 time .006 episode .004 example .006 important .004 major .004 new .006 group .004
          song .004 album .004 ship .004 storm .004 person .004 team .004 company .004 four .004 world .004 country .004""",
    "same": """time .1 year .06 name .03 period .015 day .015 way .015 as .03 season .01 position .01 area .006 game .006
          month .006 place .006 level .006 role .006 album .004 night .006 series .004 song .004 story .004
          character .004 team .006 location .004 manner .004 one .006 reason .004 date .004 style .004 week .006
          thing .004 type .004 kind .004 rank .004 route .004 basic .003""",
    "following": """the .2 year .05 day .03 season .03 his .03 a .02 their .02 her .01 its .01 week .02 month .02 morning .01
          an .006 two .006 several .005 these .005 this .006 world .006 game .006 night .006 decade .005 years .006
          summer .006 spring .005 winter .004 match .004 her .01 months .004 weeks .004 days .004 day .03 release .004""",
    "known": """as .45 for .12 to .08 from .02 in .02 by .015 that .015 about .01 only .006 species .006 and .01 of .01
          at .006 with .006 throughout .004 locally .004 examples .004 today .004 works .003 cases .003 specimens .003""",
    "called": """the .15 a .08 for .02 it .015 him .01 an .01 upon .006 on .006 up .006 them .006 his .006 her .006 this .006
          off .006 back .005 into .004 in .006 to .004 by .004 her .006""",
    "named": """after .3 the .06 for .03 in .03 by .02 as .03 to .02 a .015 him .005 best .005 one .006 his .006 her .004
          it .004 player .006 head .006 captain .005 co .005 on .006 at .006 from .006 president .004 because .004""",
    "released": """on .2 in .2 as .08 the .06 a .04 by .03 to .03 for .03 its .01 his .01 their .01 worldwide .006 through .01
          via .006 under .008 only .006 with .01 at .01 from .01 digitally .004 later .006 internationally .004
          independently .004 exclusively .004 two .006 three .004 four .003 an .006 her .004 it .004 both .004
          separately .003 commercially .003 officially .003 shortly .003""",
    "received": """a .15 the .08 positive .06 his .03 an .03 mixed .03 generally .03 critical .02 favorable .015 by .02 from .02
          its .015 their .01 more .01 widespread .01 her .01 praise .01 considerable .008 numerous .006 two .006
          several .006 high .006 little .006 much .006 over .006 universal .006 acclaim .006 negative .006 awards .004
          favourable .006 international .004 three .004 national .004 strong .004 heavy .003 only .004 no .004""",
    "began": """to .3 the .06 in .05 a .03 his .03 on .03 work .03 their .02 her .01 as .02 at .02 with .02 working .02
          construction .01 playing .01 recording .01 production .01 filming .01 its .01 writing .006 an .006
          attacking .004 searching .004 moving .004 using .004 developing .004 shortly .004 when .004 after .004
          during .004 an .006 training .004 talks .003 operations .003""",
    "became": """the .15 a .12 one .03 an .03 more .02 his .02 involved .02 known .02 part .015 increasingly .01 president .01
          its .01 their .01 first .01 popular .01 king .008 very .008 her .006 pregnant .004 close .006 friends .006
          available .006 extinct .004 head .004 less .004 much .004 chief .004 too .004 ill .004 one .03 clear .006
          interested .006 apparent .006 famous .004 so .004 known .02 associated .004 aware .004""",
    "took": """place .15 the .1 part .08 over .06 a .05 his .03 him .02 her .02 their .02 up .02 control .02 office .01
          command .01 it .01 them .01 an .008 advantage .008 on .01 off .008 two .006 three .005 four .004 out .006
          time .006 refuge .004 possession .004 charge .006 back .004 home .004 its .006 another .004 note .003""",
    "made": """a .1 the .08 his .1 by .04 of .03 her .03 up .03 in .02 it .03 an .02 their .02 to .015 from .015 its .01
          him .01 no .01 two .01 several .008 public .006 available .006 them .006 into .008 for .008 with .008 at .008
          on .008 as .006 landfall .01 changes .006 contact .006 appearances .006 use .006 major .004 significant .004
          history .004 three .004 four .003 her .03 only .004 headlines .003 sure .004 clear .004""",
    "used": """to .2 in .12 as .08 by .06 for .06 the .04 a .03 on .02 it .01 them .006 his .006 this .006 only .006
          primarily .006 mainly .004 both .004 extensively .004 throughout .004 during .006 against .006 until .006
          at .01 with .01 from .01 an .006 widely .003 again .003 instead .003 several .003""",
    "based": """on .55 in .15 at .05 upon .02 out .02 around .01 near .006 largely .006 solely .004 loosely .004 primarily .004
          partly .004 from .006 the .01 off .004 heavily .004 mainly .004""",
    "located": """in .4 on .2 at .1 near .05 the .02 approximately .02 about .02 within .02 just .01 between .01 along .01
          north .01 south .01 east .01 west .01 adjacent .006 inside .004 outside .004 off .004 across .004 above .003
          below .003 beneath .003 northeast .004 northwest .004 southeast .004 southwest .004""",
    "born": """in .6 on .15 to .05 at .03 and .02 into .01 the .01 c .003 as .006 with .004 out .003 from .003""",
    "died": """in .3 on .15 of .1 at .05 the .02 from .03 after .03 and .03 before .01 while .01 during .01 aged .01 two .006
          without .006 a .01 when .006 shortly .006 three .004 young .003 suddenly .003 just .003 only .003 later .003""",
    "won": """the .3 a .05 his .03 their .03 an .02 two .02 three .015 all .01 by .03 it .01 her .01 five .006 four .008
          both .006 several .006 only .006 best .008 first .006 back .004 its .006 every .004 one .006 six .004
          more .004 seven .003 eight .003 nine .003 games .004 matches .004""",
    "played": """in .15 the .1 a .08 by .05 for .06 as .04 his .04 at .03 on .03 with .03 an .02 two .02 their .01
          against .03 only .008 football .006 college .006 professional .004 there .004 well .004 three .006 four .004
          all .006 several .004 under .004 alongside .004 basketball .004 major .004 him .004 her .004 out .004
          one .004 every .004 five .004 baseball .004 cricket .004 rugby .004 hockey .004""",
    "game": """was .04 and .03 the .02 in .03 is .03 of .02 against .02 as .01 for .02 to .02 with .02 that .01 has .01
          had .01 boy .006 series .006 play .006 design .006 director .005 engine .004 were .004 which .008 at .008
          on .008 by .006 from .006 ended .004 began .004 features .004 takes .004 received .004 won .004 lost .004
          but .006 when .004 where .004 despite .004 world .004 he .004 they .004 it .004 its .003 itself .003""",
    "season": """and .03 the .02 in .03 with .04 was .03 he .02 of .02 for .02 as .02 at .02 to .02 finale .008 premiere .006
          began .006 ended .006 they .006 she .004 after .006 before .004 when .004 which .008 had .006 has .006 is .006
          on .006 by .006 from .006 but .006 were .004 opener .004 record .004 high .004 ticket .003 two .004 three .004
          four .004 five .004 six .003 seven .003 eight .003 nine .003 one .004 average .003""",
    "film": """was .05 and .03 the .02 in .03 is .03 of .02 as .01 to .02 director .01 festival .01 received .006 has .008
          had .006 which .008 that .008 with .01 by .008 for .01 on .008 at .008 from .006 industry .004 adaptation .006
          version .006 debut .006 score .004 series .004 critics .004 grossed .004 stars .004 where .004 but .006
          when .004 premiered .004 opened .004 earned .004 itself .003 rights .003 starring .004 about .004 he .004""",
    "album": """was .05 the .03 and .03 in .03 is .02 of .02 chart .015 charts .01 with .02 which .01 debuted .01 peaked .008
          received .006 has .006 had .006 that .008 by .006 for .01 on .01 at .006 from .006 as .01 to .01 were .004
          sold .006 includes .004 features .004 title .004 cover .004 tracks .004 producer .004 but .006 when .004
          reached .004 contains .004 itself .003 he .004 she .004 they .004 release .004 sales .004 artwork .003""",
    "song": """was .05 and .03 the .02 in .03 is .03 of .02 has .008 with .02 as .01 to .02 for .01 on .01 by .01 that .01
          which .01 received .006 peaked .008 debuted .008 lyrics .006 features .006 were .004 itself .004 title .004
          from .01 at .008 but .006 when .004 reached .006 became .004 he .004 she .004 they .004 also .004
          entered .004 charted .004 performed .004 premiered .003 opens .003 begins .003""",
    "city": """of .15 and .03 in .03 council .03 the .02 was .02 is .02 to .02 centre .01 center .01 hall .01 with .01
          for .01 that .008 which .008 has .008 had .006 limits .006 walls .006 from .008 on .008 at .008 by .006
          as .006 where .006 fc .006 football .004 were .004 but .004 when .004 itself .003 government .004
          police .004 officials .004 state .004 park .004 st .003 university .004""",
    "river": """and .03 in .03 the .02 to .03 valley .03 at .02 which .015 is .015 was .01 with .01 from .015 near .01
          basin .01 for .006 on .01 by .006 as .006 where .006 that .006 flows .006 into .006 has .006 had .004
          mouth .004 crossing .006 were .004 bridge .006 between .006 delta .004 bank .004 banks .004 runs .004""",
    "church": """of .15 and .03 in .03 the .02 is .02 was .03 which .01 has .01 at .01 to .015 with .01 for .008 on .008
          by .006 as .006 that .008 from .006 had .006 where .006 were .004 building .004 hall .004 tower .004
          yard .004 itself .003 leaders .003 authorities .003 music .003 st .003 street .003""",
    "school": """of .05 and .03 in .03 the .02 was .02 is .02 at .02 to .02 district .01 with .01 for .01 where .01 which .008
          football .006 students .006 has .006 had .006 on .008 by .006 as .006 that .006 from .006 board .004
          teacher .004 he .004 she .004 were .004 year .004 years .004 team .004 building .004 system .004""",
    "ship": """was .06 and .03 the .02 in .03 to .03 had .02 is .01 of .02 with .02 for .01 as .01 which .01 on .01 at .01
          from .01 by .008 that .008 has .006 were .004 sank .006 returned .006 arrived .006 departed .006 sailed .006
          remained .004 began .004 but .006 when .004 s .003 itself .003 class .004 design .004""",
    "storm": """was .04 and .03 the .02 in .03 to .03 moved .02 turned .01 had .01 is .006 of .02 with .01 for .01 as .02
          which .01 on .01 at .01 from .01 by .01 that .01 has .006 were .004 caused .01 made .01 dropped .006
          produced .01 weakened .01 intensified .01 continued .01 began .006 crossed .006 struck .006 tracked .006
          brought .006 killed .006 while .006 when .006 after .006 before .004 surge .008 warning .006 watch .006
          but .006 executed .003 passed .004 approached .004 attained .004 reached .004 quickly .004 rapidly .004""",
    "tropical": """storm .5 depression .2 cyclone .1 storms .06 cyclones .03 forests .006 forest .006 regions .004 waves .004
          wave .006 moist .004 and .004 rainforest .004 rainforests .004 climate .004 areas .004 islands .003
          disturbance .006 system .004 species .003""",
    "hurricane": """was .03 and .03 the .02 in .02 season .03 to .02 of .01 status .03 watch .02 warning .02 warnings .01
          force .01 center .006 hunters .006 with .01 intensity .01 katrina .01 had .01 is .006 for .006 on .006
          at .006 as .006 that .004 which .004 hit .004 made .004 struck .004 caused .004 moved .004""",
    "army": """and .04 in .03 the .02 of .04 was .02 to .03 had .02 under .02 at .02 from .02 with .02 for .01 on .01 as .01
          which .01 that .01 by .01 officer .01 officers .01 group .02 corps .02 units .01 were .006 air .015 forces .006
          general .006 headquarters .006 command .006 division .006 reserve .006 but .006 when .004 is .006 has .004
          commander .004 began .004 attacked .004 advanced .004 moved .004 crossed .004 retreated .004""",
    "battle": """of .5 and .03 in .03 was .02 with .02 to .02 at .02 for .01 on .01 as .01 that .01 which .01 honours .006
          honour .004 cruiser .006 cruisers .006 group .004 squadron .006 fleet .006 line .004 from .006 by .006
          between .006 against .006 near .006 ended .004 began .004 he .004 the .006 is .004 has .004 had .004""",
    "division": """and .04 in .03 the .02 was .03 to .03 of .03 with .02 at .02 had .02 from .02 for .01 on .01 as .01
          which .01 that .01 by .01 were .004 is .006 commander .01 headquarters .01 title .01 one .01 two .008
          three .006 championship .006 but .006 when .004 moved .006 began .006 attacked .006 advanced .006
          launched .004 remained .004 suffered .004 football .004 side .004 club .004 status .004""",
    "census": """the .25 there .2 of .03 and .03 in .03 was .02 to .02 data .02 bureau .02 estimates .02 estimate .01
          designated .01 population .02 records .01 showed .01 had .01 returns .006 tract .004 figures .006 it .02
          its .01 this .01 a .01 an .006 which .01 taken .006 recorded .006""",
    "population": """of .45 was .05 and .03 in .03 is .03 to .02 density .03 growth .01 had .01 has .01 at .01 for .01 with .01
          as .01 on .01 from .006 by .006 were .006 that .006 which .006 decline .006 declined .006 increased .006
          grew .006 rose .006 reached .006 fell .004 centres .004 but .006 estimate .004 estimates .004 figures .004""",
    "species": """of .15 is .04 and .03 in .03 was .02 are .02 to .02 have .01 has .01 that .01 which .01 with .02 for .01
          on .01 at .006 from .01 by .01 as .01 were .01 can .006 such .006 including .006 found .006 known .006
          name .01 description .006 but .006 had .006 within .006 occur .004 may .004 belong .004 described .004""",
    "prime": "minister .85 time .03 example .01 number .01 ministers .02 meridian .004 location .004 target .004",
    "minister": """of .3 and .03 for .05 in .03 was .02 who .02 to .02 at .01 with .01 had .01 the .01 john .006
          david .004 he .004 is .006 has .004 his .004 from .004 on .004 by .004 as .004""",
    "los": "angeles .95",
    "san": """francisco .35 diego .2 antonio .1 jose .06 juan .05 marino .03 luis .02 andreas .02 fernando .02 pedro .02
          salvador .02 miguel .01 bernardino .01 gabriel .01 mateo .01 sebastian .01""",
    "hong": "kong .95",
    "sri": "lanka .9 lankan .05",
    "air": """force .35 forces .05 raid .03 raids .02 base .03 support .02 attack .02 attacks .02 station .02 defense .02
          defence .02 service .02 power .01 to .01 in .01 date .01 temperature .006 and .02 corps .02 strikes .01
          traffic .006 quality .004 national .004 cover .006 superiority .004 squadron .004 group .004 arm .004""",
    "royal": """navy .25 air .1 family .05 society .04 academy .03 marines .02 australian .03 canadian .02 artillery .02 army .02
          engineers .015 court .015 house .015 palace .01 college .015 albert .01 opera .01 highness .01 charter .01
          assent .006 irish .006 scots .006 welsh .006 household .006 flying .01 naval .01 regiment .01 infantry .006
          commission .006 marine .004 dutch .004 danish .004 italian .004 swedish .004 hospital .004 mail .004""",
    "high": """school .3 and .02 level .02 court .02 quality .02 speed .02 point .015 street .01 end .01 density .006 profile .006
          ground .006 winds .01 water .006 rate .006 seas .004 pressure .006 commission .004 priest .004 sheriff .004
          temperatures .004 schools .02 in .01 on .006 as .006 enough .004 above .004 levels .004 praise .004""",
    "video": """game .35 games .15 for .03 was .02 and .02 of .02 in .02 the .01 directed .03 release .01 features .01
          shows .01 music .01 premiered .01 clip .006 footage .006 is .01 also .006 to .01 with .01 on .006 at .006
          by .006 from .006 which .006 that .006 began .004 starts .004 opens .004 begins .004""",
    "music": """video .15 and .04 of .03 in .03 is .01 was .01 for .02 videos .03 industry .02 critics .02 festival .01 hall .01
          chart .01 charts .01 scene .01 producer .006 with .01 by .01 from .01 as .006 to .01 at .006 awards .006
          director .004 school .004 that .006 which .006 has .006 the .006 critic .006 journalist .004 composed .004""",
    "billboard": """hot .4 charts .05 magazine .04 and .02 chart .03 music .02 year .01 pop .02 adult .02 dance .02 japan .01
          top .01 alternative .01 mainstream .01 rock .01 latin .01 country .01 the .006 also .006""",
    "later": """became .05 the .05 in .04 that .02 he .02 years .03 renamed .02 on .03 released .02 known .01 moved .01 wrote .01
          said .01 stated .01 joined .01 returned .01 served .01 recalled .01 described .006 admitted .006 used .006
          appeared .006 life .02 career .01 date .01 period .006 to .01 at .01 by .01 with .01 as .01 work .006
          version .006 a .01 an .006 that .02 years .03 she .01 they .01 it .01 his .006 her .006 two .004 three .004
          became .05 reported .006 explained .006 claimed .006 founded .004 established .004""",
    "early": """years .05 life .03 in .06 on .03 career .02 stages .01 as .03 part .02 twentieth .01 nineteenth .01 morning .02
          hours .01 history .01 september .006 october .006 and .02 to .02 for .01 of .01 work .01 days .01 period .01
          modern .01 development .006 medieval .006 age .006 season .006 next .006 warning .004 twenty .006 1990s .004
          august .006 july .006 june .006 may .004 april .004 march .004 february .004 january .004 november .004
          december .004 christian .006 american .004 english .004 years .05 spring .004 summer .004 autumn .004
          winter .004 evening .004 afternoon .004 education .004 adulthood .003 childhood .004""",
    "north": """of .15 america .08 american .05 carolina .06 korea .03 korean .02 and .03 to .03 side .02 east .02 west .02
          end .01 atlantic .02 sea .015 africa .015 london .006 wales .006 dakota .015 bay .006 island .006 coast .01
          the .01 into .006 along .006 through .006 bank .004 shore .004 face .004 tower .004 wall .004 yorkshire .004""",
    "south": """of .12 africa .07 carolina .05 korea .03 america .03 american .02 wales .02 and .03 to .03 east .02 west .02
          side .02 end .01 dakota .015 australia .015 pacific .015 island .006 sea .006 bank .004 african .015
          korean .01 china .01 asia .006 coast .006 the .01 into .006 along .006 through .006 park .004 london .004
          vietnam .006 vietnamese .004 yorkshire .004 shore .004 street .004""",
    "east": """of .15 and .03 to .03 side .02 end .02 coast .02 germany .015 asia .015 india .01 africa .01 anglia .006
          timor .006 london .006 bank .006 river .006 midlands .004 the .01 into .006 along .006 through .006
          german .006 african .006 indian .004 asian .004 sussex .004 street .004 wing .004 tower .004""",
    "west": """of .15 and .03 to .03 side .02 coast .02 end .01 virginia .03 indies .02 germany .02 african .01 africa .015
          bank .01 point .006 midlands .006 ham .006 bromwich .006 london .006 the .01 into .006 along .006
          through .006 german .006 indian .004 coast .02 country .004 street .004 wing .004 tower .004 front .004""",
    "national": """park .05 team .04 league .04 football .03 guard .03 register .03 historic .03 army .02 assembly .02
          championship .015 anthem .01 party .015 government .01 level .01 and .02 hockey .015 basketball .006
          museum .015 library .01 trust .01 front .006 security .006 defense .006 theatre .006 council .01 parks .006
          monument .008 championships .006 hurricane .015 weather .015 highway .01 road .006 health .006 service .006
          socialist .004 academy .006 institute .004 of .01 in .008 squad .004 teams .004 side .004 cup .004
          title .006 natural .004 forest .004 radio .004 public .004 television .004 film .004 gallery .004""",
    "football": """league .15 club .1 team .05 and .02 in .02 at .02 player .02 association .03 championship .01 conference .01
          ground .01 for .01 manager .006 teams .01 coach .01 program .01 season .01 matches .006 match .006 game .006
          games .006 stadium .006 players .006 in .02 history .004 career .004 field .004 was .004 is .004""",
    "league": """and .04 the .02 in .03 baseball .05 cup .04 championship .02 season .03 title .02 with .02 for .02 one .02
          two .015 games .02 matches .01 debut .015 of .02 to .02 football .01 was .01 is .01 as .01 at .01 on .01
          from .006 by .006 level .006 where .006 which .006 that .006 career .006 history .006 goals .006 rivals .004
          but .006 teams .006 appearances .006 leaders .004 record .004 table .004 trophy .004 club .004""",
    "club": """and .04 in .03 the .02 for .03 was .02 is .01 to .02 of .02 with .02 football .01 at .01 history .01 record .01
          captain .01 president .006 which .01 that .01 where .006 on .01 from .01 by .006 as .01 had .006 has .006
          were .004 but .006 level .004 s .003 he .004 they .004 members .004 officials .004 secretary .004""",
    "team": """the .02 and .03 in .03 was .02 to .03 of .02 with .02 for .02 that .01 as .01 won .01 had .01 finished .01
          which .01 at .01 on .01 from .01 by .006 is .01 has .006 were .004 lost .006 played .006 went .006 also .006
          captain .006 but .006 when .004 record .006 history .006 members .006 leader .004 mates .004 also .006
          scored .004 began .004 made .004 qualified .004 reached .004 failed .004 he .004 they .004 s .003""",
    "km": """long .05 north .03 south .03 east .03 west .03 from .03 away .02 of .04 to .02 in .03 and .03 per .01 northwest .01
          northeast .01 southwest .01 southeast .01 wide .02 high .01 east .03 off .01 inland .01 offshore .006 at .006
          the .006 or .006 northward .004 southward .004 west .03 deep .006 square .004""",
    "mi": """long .03 north .02 south .02 east .02 west .02 from .02 away .02 of .03 to .02 in .02 and .03 per .01 wide .01
          northwest .01 northeast .01 southwest .01 southeast .01 off .006 inland .006 offshore .006 the .006""",
    "per": "cent .35 capita .2 game .08 year .05 day .03 hour .03 second .02 week .02 month .02 season .02 person .02 episode .01 share .01",
    # —— 两个词的上下文 ——
    "one of": "the .75 his .04 their .03 its .03 her .01 two .01 these .01 those .01 only .006 many .006 several .006",
    "as well": "as .9 known .01 received .01",
    "in order": "to .9 for .02 of .01",
    "a number": "of .95",
    "the end": "of .85 the .01",
    "the rest": "of .9",
    "the age": "of .9",
    "at least": """one .06 two .04 three .02 the .03 a .03 some .01 in .02 part .01 partially .01 until .01 partly .006
          four .01 five .01 for .01 to .01 six .006 ten .006 as .01 that .006 an .006 his .006 one .06 twenty .004""",
    "the united": "states .7 kingdom .2 nations .07",
    "united states": """navy .05 army .04 and .05 in .05 to .03 government .02 census .02 house .02 senate .02 military .02 marine .02
          air .02 department .015 congress .015 district .015 for .02 on .02 with .02 by .02 as .02 from .02 at .015
          that .015 was .01 is .01 open .006 court .006 national .006 forces .006 where .006 which .006""",
    "world war": "ii .55 i .3",
    "new york": """city .25 times .08 and .03 in .03 to .02 yankees .02 giants .015 state .015 jets .006 mets .006 rangers .006
          knicks .006 islanders .004 post .01 university .006 stock .006 harbor .006 where .006 with .01 for .01 on .01
          at .01 from .01 by .01 as .01 was .006 is .006 after .006 public .004 daily .004""",
    "it was": """the .06 a .07 not .05 also .04 released .04 an .02 announced .03 one .02 later .02 used .02 built .02 in .02
          only .01 first .01 decided .01 renamed .01 originally .01 replaced .01 reported .01 named .01 designed .01
          very .006 his .006 her .006 no .006 revealed .006 confirmed .006 discovered .006 still .006 believed .006
          made .006 sold .006 given .006 written .006 recorded .006 considered .006 described .006 found .006""",
    "there was": """a .2 no .1 an .04 little .03 the .02 also .03 some .02 not .02 only .01 much .01 one .01 nothing .006
          considerable .006 significant .006 widespread .006 minimal .006 never .006 still .006 another .006 more .006
          heavy .004 extensive .004 great .004 strong .004 limited .004""",
    "there were": """no .1 a .04 also .04 two .03 many .03 several .03 some .03 three .02 only .02 more .02 over .02 about .02
          approximately .02 still .01 few .01 reports .01 plans .01 other .01 four .006 five .006 six .004 ten .004
          numerous .006 concerns .006 rumors .004 rumours .004 an .004 at .006 between .004 nearly .006 almost .006""",
    "there are": """no .08 two .05 many .04 several .04 also .04 three .03 some .03 a .02 more .02 other .02 over .02 about .02
          approximately .02 only .02 few .01 still .01 four .01 currently .01 five .006 numerous .006 at .006 an .004""",
    "he was": """a .06 the .05 also .03 not .03 born .03 appointed .02 elected .02 promoted .02 named .02 then .015
          awarded .015 replaced .01 given .01 sent .01 killed .01 able .01 made .01 released .006 in .02 at .01 an .01
          one .01 only .01 still .01 later .01 unable .008 eventually .008 considered .008 very .006 buried .006
          posted .006 transferred .006 assigned .006 selected .006 signed .006 described .006 known .006 forced .006
          married .004 captured .004 wounded .004 educated .006 raised .004 succeeded .004 called .004 placed .004""",
    "she was": """a .06 the .05 also .03 not .03 born .03 launched .02 laid .02 decommissioned .01 commissioned .01 sold .01
          assigned .01 struck .01 sunk .006 sent .006 placed .006 transferred .006 renamed .006 named .006 in .02
          at .01 an .008 one .008 only .008 still .008 towed .004 ordered .004 refitted .004 converted .004
          scrapped .004 reduced .004 married .004 awarded .004 given .004 able .004 then .006 later .006""",
    "was released": """on .25 in .2 as .12 the .03 a .03 to .03 by .02 for .03 through .02 worldwide .01 digitally .01 via .01
          under .01 with .01 from .01 at .01 only .01 later .006 internationally .006 physically .004 exclusively .006
          two .004 three .004 four .003 several .003 one .004 an .006 its .004 his .004 her .004 their .004""",
    "was born": "in .6 on .2 to .08 at .03 into .02 and .02",
    "has been": """a .06 the .04 described .03 used .03 called .02 an .015 made .015 in .02 released .015 named .01 noted .01
          compared .01 credited .01 praised .01 criticized .01 criticised .006 nominated .01 suggested .01 involved .01
          at .01 on .01 one .01 very .006 more .006 considered .01 identified .01 recorded .01 found .01 reported .01
          proposed .006 covered .006 performed .006 featured .006 listed .006 seen .006 cited .006 designated .006""",
    "had been": """a .06 the .04 in .03 killed .02 made .015 an .01 used .015 built .01 given .01 destroyed .01 planned .01
          damaged .01 sent .01 promoted .006 released .006 able .006 on .01 at .01 with .006 for .006 very .006
          more .006 completed .006 captured .006 working .006 playing .004 living .004 taken .006 appointed .006
          replaced .006 assigned .006 involved .006 named .006 established .006 married .006 reduced .004""",
    "have been": """a .05 the .04 used .03 made .02 found .02 described .02 an .01 in .02 suggested .015 identified .015
          recorded .015 proposed .01 reported .01 known .01 able .01 built .01 killed .01 published .01 released .01
          given .01 named .01 at .01 on .01 more .006 very .006 several .006 many .006 no .006 written .006""",
    "the first": """time .05 season .03 game .03 two .02 world .015 album .015 single .015 film .012 half .012 episode .012
          three .01 year .01 day .01 round .01 week .008 major .008 one .006 class .008 to .02 of .01 in .01 and .01
          person .006 woman .006 president .004 female .004 edition .006 part .006 book .006 volume .004 issue .006
          quarter .006 inning .004 battle .006 british .004 american .004 national .004 full .006 public .006
          official .004 known .004 recorded .004 ever .004 lady .004 phase .004 stage .004 flight .004 night .004
          four .004 five .004 six .003 ten .003 such .004 being .003 world .015 black .003 african .003""",
    "the same": """time .1 year .06 name .03 period .015 day .015 way .015 season .01 position .01 area .006 game .006
          month .006 night .006 place .006 level .006 role .006 album .004 series .004 song .004 story .004
          character .004 team .006 location .004 manner .004 one .006 reason .004 date .004 style .004 as .03 week .006
          thing .004 type .004 kind .004 rank .004 route .004 basic .003 building .003 people .003 general .003""",
    "the following": """year .15 season .08 day .05 month .04 week .03 morning .03 years .01 night .02 game .02 decade .01
          two .01 summer .01 spring .01 winter .006 january .006 autumn .006 april .006 is .01 are .01 episode .006
          match .006 list .004 table .004 seasons .006 months .006 weeks .006 days .006 decades .004 races .004""",
    "the most": """important .04 popular .03 common .02 successful .02 notable .02 famous .015 recent .01 significant .01
          powerful .008 prominent .008 part .01 widely .006 commonly .006 influential .006 valuable .006 difficult .006
          expensive .006 likely .006 intense .006 frequently .006 highly .006 prolific .004 severe .004
          destructive .004 damaging .004 active .004 well .004 successful .02 critically .004 recognizable .004
          complete .004 extensive .004 northerly .003 southerly .003 westerly .003 easterly .003 dangerous .004""",
    "part of": """the .45 a .06 his .04 their .03 its .03 an .02 this .01 what .01 her .01 it .01 which .01 both .006 new .006
          that .006 these .006 many .004 two .004 operation .006 british .004 greater .006 western .004 eastern .004
          northern .004 southern .004 one .004 several .004 each .004 our .003""",
    "such as": """the .15 a .04 his .01 those .01 these .006 this .006 their .01 its .006 an .01 in .006 when .006 how .006
          being .006 new .004 john .003 william .003 george .003 england .003 france .003 germany .003 london .003""",
    "due to": """the .35 a .05 his .03 its .03 their .03 an .01 poor .01 injury .01 injuries .006 lack .01 financial .006
          budget .006 increased .006 low .006 high .006 heavy .006 strong .006 concerns .006 being .006 her .006
          this .01 these .006 time .006 illness .006 health .006 problems .006 wind .006 rain .006 flooding .006
          conflicts .004 scheduling .004 other .004 its .03 copyright .003 limited .004 rising .004 changes .004""",
    "according to": """the .35 a .05 his .03 their .02 some .01 its .015 her .01 one .01 nielsen .02 official .006 local .006
          historian .006 legend .02 tradition .01 reports .006 sources .006 him .006 them .006 data .006 an .006
          estimates .006 rolling .004 billboard .004 allmusic .004 historians .004 scholars .004 several .004 media .004
          witnesses .004 contemporary .004 later .004 most .004 many .004 author .004 other .004 this .004""",
    "in addition": "to .5 the .1 he .02 it .01 a .01 there .01 his .006 two .006 several .006",
    "as a": """result .1 member .04 single .03 child .03 player .02 whole .02 means .01 consequence .01 young .02 teenager .01
          writer .01 freshman .01 junior .006 senior .006 sophomore .006 professional .006 free .006 student .01
          way .01 form .006 symbol .006 major .006 new .006 small .006 guest .006 replacement .006 reserve .006
          training .006 solo .006 regular .006 starter .006 tropical .006 boy .006 girl .004 director .006
          producer .006 composer .004 singer .006 teacher .004 soldier .004 lieutenant .004 captain .004 private .004
          gift .004 tribute .004 sign .004 reward .004 punishment .004 promotional .004 bonus .004 kind .004""",
    "as a result": "of .4 the .1 he .02 it .01 a .01 they .01 she .006 there .01 many .006 his .006",
    "at the time": """of .25 the .05 he .03 and .03 was .02 it .02 she .01 they .01 a .01 had .006 in .01 but .01 as .01
          that .01 when .01 said .006 his .006 her .006 its .006 their .006 being .006""",
    "one of the": """most .1 first .05 best .04 largest .03 two .02 few .02 main .015 greatest .015 major .015 highest .01
          earliest .01 last .01 oldest .01 biggest .01 top .01 longest .006 three .01 key .006 leading .006
          founding .006 only .006 strongest .006 worst .006 early .006 finest .006 original .006 great .006
          more .004 four .004 five .004 deadliest .004 fastest .004 smallest .004 richest .004 youngest .004
          founders .004 primary .004 central .004 principal .004 important .004 world .004 band .004""",
    "the end of": """the .5 his .05 a .03 their .02 its .02 world .03 her .01 this .01 that .01 each .006 june .006 july .006
          august .006 september .006 october .006 november .006 december .006 january .006 february .006 march .006
          april .006 may .004 season .006 an .004 one .004 two .003 three .003 every .003 year .004""",
    "a number of": """other .03 different .02 songs .01 new .01 times .01 people .01 species .01 small .01 british .006 works .006
          local .006 smaller .006 changes .006 years .006 occasions .006 areas .006 factors .004 films .004
          notable .004 important .004 critics .004 reviewers .004 ships .004 players .004 countries .004 cities .004
          sources .004 books .004 tracks .004 shows .004 major .004 key .004 large .004 high .004""",
}
MAX_LISTED = 0.95  # 一张表里列出的概率之和最多占这么多，剩下的留给语法头兜底

# 日期：维基百科常写 "14 September 1940"、"in May 1990"、"on May 5"
MONTHS = "january february march april may june july august september october november december".split()
for _key, _p in (("<num>", 0.012), ("in", 0.003), ("on", 0.003)):
    NGRAMS[_key] += " " + " ".join(f"{m} {_p}" for m in MONTHS)

# 固定介词搭配：这些词后面很常跟某个介词（只补 NGRAMS 里还没有的词）
PREP_COLLOCATIONS = {
    "to": """led related similar compared returned moved went sent assigned attached added referred close next prior
          contrary access response return reference contribution moving going travelled traveled transferred
          promoted relegated elected appointed sold given lent loaned owing thanks applied opposed devoted
          dedicated committed limited restricted adjacent equivalent subject exposed listening tribute""",
    "of": """consisting consists capable outside front top side kind type lack loss majority source percent instead
          member members none use form series range variety aware edge bottom middle center centre beginning
          start course capital site position head chief director president mayor governor duke earl university
          department ministry secretary period total version cause history remains death birth town city
          village county state province kingdom emperor king queen prince sort""",
    "by": """followed caused written directed produced replaced owned operated surrounded killed published defeated
          designed founded commissioned composed performed voiced influenced inspired accompanied characterized
          characterised dominated preceded hosted organized organised sponsored funded authored illustrated
          edited narrated co""",
    "with": """along together associated compared filled equipped dealing dealt contact combined armed charged faced
          coupled credited familiar concerned satisfied impressed collaborated partnered merged clashed tied
          fitted dealing interview relationship relations""",
    "in": """interest involved born participated resulted increase decline interested engaged specialized
          specialised situated settled stationed buried enrolled imprisoned starring""",
    "from": "away apart derived far suffering retired resigned graduated recovered emerged originated escaped",
    "on": "focused focusing depend depends depending relied relying effect impact emphasis",
    "for": "responsible famous reason nominated candidate search credited prepared eligible renowned",
    "into": "divided converted turned split incorporated integrated transformed",
    "as": "regarded acted served doubled",
}
PREP_COLLOCATION_P = 0.3

# ═══════════════════════ 第 4、5 层：复制头、归纳头 ═══════════════════════

LAMBDA_CACHE = 0.10          # 复制头：段落里出现过的实词（不含功能词、不含紧挨着的前一个词）
CACHE_HALF = 10              # 复制头权重 × n/(n+CACHE_HALF)：上下文里实词少时自动减弱
LAMBDA_IND = 0.20            # 归纳头：上次"前一个词"后面跟的词
IND_FUNCTION_FACTOR = 0.25   # 前一个词是功能词时（后面跟什么都有可能），归纳头打这个折扣
IND_DET_FACTOR = 0.6         # 前一个词是限定词（the / his …）时的折扣：后面多半是同一批话题名词
IND_BIGRAM_WEIGHT = 3.0      # 前两个词都对上时的权重


# ═══════════════════════ 下面是把上面的手写知识组装成权重的代码 ═══════════════════════

def _closed_map():
    m = {}
    for cls, text in CLOSED.items():
        for w in text.split():
            m.setdefault(w, set()).add(cls)
    for w, extra in CLOSED_EXTRA.items():
        m.setdefault(w, set()).update(extra)
    return {w: tuple(sorted(s)) for w, s in m.items()}


CLOSED_MAP = _closed_map()
ADJ_SET = set(ADJ_WORDS.split())
VB_SET = set(VB_WORDS.split())
NV_SET = set(NV_WORDS.split())
VBD_SET = set(VBD_WORDS.split())
VERBISH = VB_SET | NV_SET


@lru_cache(maxsize=None)
def classify(w):
    """一个词可能的词类（取多个时，作为候选词按其中最有利的那个算）。"""
    if w in CLOSED_MAP:
        return CLOSED_MAP[w]
    cls = set()
    if w in ADJ_SET:
        cls.add("ADJ")
    if w in VB_SET:
        cls.add("VB")
    if w in NV_SET:
        cls.update(("NN", "VB"))
    if w in VBD_SET:
        cls.add("VBD")
    if cls:
        return tuple(sorted(cls))
    n = len(w)
    if n > 4 and w.endswith("ly") and w not in LY_NOT_ADV:
        return ("ADV",)
    if n > 4 and w.endswith("ing") and w not in ING_NOT_VBG:
        return ("NN", "VBG")
    if n > 4 and w.endswith("ed") and w not in ED_NOT_VBD:
        return ("ADJ", "VBD")
    if n > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ics")):
        stem = w[:-2] if w.endswith("es") and w[:-2] in VERBISH else w[:-1]
        return ("NNS", "VBZ") if stem in VERBISH else ("NNS",)
    if n > 5 and w.endswith(("ous", "ful", "ive", "able", "ible", "less", "ish", "ical")):
        return ("ADJ",)
    if n > 4 and w.endswith(("al", "ic", "ent", "ant", "ary", "ian", "ean", "can", "ese", "ist")):
        return ("ADJ", "NN")
    return ("NN",)


def norm_token(t):
    if any(ch.isdigit() for ch in t):
        return "<year>" if YEAR_RE.match(t) else "<num>"
    return t


def token_classes(t):
    """上下文里一个 token 的类别（标点、数字也算）。"""
    if t in (".", "!", "?"):
        return ("START",)
    if t == ",":
        return ("COMMA",)
    if t in ("(", "["):
        return ("OPEN",)
    if t in (")", "]"):
        return ("CLOSE",)
    if t == '"':
        return ("QUOTE",)
    if t in (";", ":"):
        return ("COLON",)
    if t in ("-", "–", "—"):
        return ("DASH",)
    if t in ("'s", "'", "s'"):
        return ("POSSM",)
    if t == "n't":
        return ("ADV",)
    if t == "<year>":
        return ("YEAR",)
    if t == "<num>":
        return ("NUM",)
    if WORD_RE.match(t):
        return classify(t)
    return ("OTHER",)


def vowel_sound(w):
    if w.startswith(("uni", "use", "usu", "uti", "ura", "eu", "one", "once")):
        return False
    if w.startswith(("hour", "honest", "honor", "honour", "heir")):
        return True
    return w[0] in "aeiou"


def derived_zipf(w, listed):
    """没写到的词：从写到的词根 / 变形推一个 Zipf 值。"""
    found = []

    def add(stem, drop):  # 功能词和太短的词不当词根（否则 his→hi、and→andes、in→inning）
        if stem in listed and len(stem) >= 3 and stem not in CLOSED_MAP:
            found.append(listed[stem] - drop)

    n = len(w)
    if n > 3 and w.endswith("s") and not w.endswith("ss"):
        add(w[:-1], 1.2 if w[:-1] in ADJ_SET else 0.35)  # 形容词很少加 s（firsts）
        if w.endswith("es"):
            add(w[:-2], 0.4)
        if w.endswith("ies"):
            add(w[:-3] + "y", 0.4)
    if n > 4 and w.endswith("ed"):
        add(w[:-2], 0.4)
        add(w[:-1], 0.4)
        if w[-3] == w[-4]:
            add(w[:-3], 0.4)
        if w.endswith("ied"):
            add(w[:-3] + "y", 0.4)
    if n > 5 and w.endswith("ing"):
        add(w[:-3], 0.5)
        add(w[:-3] + "e", 0.5)
        if w[-4] == w[-5]:
            add(w[:-4], 0.5)
    if n > 4 and w.endswith("ly"):
        add(w[:-2], 0.7)
        add(w[:-1] + "e", 0.7)
        if w.endswith("ily"):
            add(w[:-3] + "y", 0.7)
    if n > 4 and w.endswith("ers"):
        add(w[:-3], 1.0)
        add(w[:-2], 1.0)
    elif n > 4 and w.endswith("er"):
        add(w[:-2], 0.8)
        add(w[:-1], 0.8)
    if n > 5 and w.endswith("est"):
        add(w[:-3], 0.9)
        add(w[:-2], 0.9)
    if n > 6 and w.endswith("ness"):
        add(w[:-4], 1.0)
    if n > 6 and w.endswith("ment"):
        add(w[:-4], 0.8)
    if n > 5 and w.endswith("tion"):
        add(w[:-3], 0.6)
        add(w[:-3] + "e", 0.6)
    for suf, drop in (("s", 0.3), ("es", 0.3), ("ed", 0.3), ("d", 0.3), ("ing", 0.4)):
        add(w + suf, drop)
    if w.endswith("e"):
        add(w[:-1] + "ing", 0.4)
    return max(found) if found else None


def estimate_zipf(w, listed):
    if w in listed:
        return listed[w]
    default = DEFAULT_ZIPF - LENGTH_SLOPE * max(0, len(w) - 6)
    d = derived_zipf(w, listed)
    return default if d is None else max(d, default)


class H2Model(ScoreModel):
    name = "H2"
    kind = "H2"
    version = VERSION
    desc = "LLM 手写权重：前缀层 + 手写词频 + 语法头 + 搭配表 + 复制头 + 归纳头（不用梯度、不读训练集）"

    def __init__(self):
        super().__init__(load_shared_vocab())
        words = self.words
        listed = {}
        for z, text in ZIPF_TIERS:
            for w in text.split():
                listed.setdefault(w, z)
        self.zipf = np.array([estimate_zipf(w, listed) for w in words])
        self.logprior = self.zipf * LN10

        sets = [classify(w) for w in words]
        self.sets = sorted(set(sets))
        sid = {s: i for i, s in enumerate(self.sets)}
        self.set_id = np.array([sid[s] for s in sets])

        vowel = np.array([vowel_sound(w) for w in words])
        self.article_pen = {"a": np.where(vowel, A_BEFORE_VOWEL, 0.0),
                            "an": np.where(vowel, 0.0, AN_BEFORE_CONSONANT)}

        self.ngrams = {}
        for key, text in NGRAMS.items():
            toks = text.split()
            table = {}
            for w, p in zip(toks[0::2], toks[1::2]):
                i = self.index.get(w)
                if i is not None:
                    table[i] = max(table.get(i, 0.0), float(p))
            if not table:
                continue
            idx = np.fromiter(table.keys(), dtype=np.int64)
            pr = np.fromiter(table.values(), dtype=np.float64)
            if pr.sum() > MAX_LISTED:
                pr *= MAX_LISTED / pr.sum()
            self.ngrams[key] = (idx, pr, float(pr.sum()))
        for prep, text in PREP_COLLOCATIONS.items():
            j = self.index.get(prep)
            for w in text.split():
                if w not in NGRAMS and w in self.index and j is not None:
                    self.ngrams[w] = (np.array([j]), np.array([PREP_COLLOCATION_P]), PREP_COLLOCATION_P)
        self._base_cache = {}

    def size_bytes(self):
        return Path(__file__).stat().st_size

    def _base(self, prev_classes, article):
        """语法头：softmax(b + T[c_prev, class(w)] + 冠词修正)；按前一个 token 的类别缓存。"""
        key = (prev_classes, article)
        p = self._base_cache.get(key)
        if p is None:
            row = {c: sum(TRANS.get(pc, {}).get(c, 0.0) for pc in prev_classes) / len(prev_classes)
                   for c in NEXT_CLASSES}
            set_scores = np.array([max(row[c] for c in s) for s in self.sets])
            s = self.logprior + set_scores[self.set_id]
            if article:
                s = s + self.article_pen[article]
            p = np.exp(s - s.max())
            p /= p.sum()
            self._base_cache[key] = p
        return p

    def context_scores(self, context):
        toks = [norm_token(t) for t in context[-3:]]
        if toks:
            prev = toks[-1]
            prev_classes = token_classes(prev)
            if prev_classes == ("ADV",) and len(toks) >= 2 and WORD_RE.match(toks[-2]):
                prev_classes = ("ADV",) + token_classes(toks[-2])  # 副词透明："were also" 按 were 的语法走
            keys = [" ".join(toks[-n:]) for n in (1, 2, 3) if len(toks) >= n]
        else:  # 段落开头
            prev, prev_classes, keys = ".", ("START",), ["."]
        article = prev if prev in ("a", "an") else None
        P = self._base(prev_classes, article).copy()

        for key in keys:  # 搭配头：短的先插值，长的后覆盖
            hit = self.ngrams.get(key)
            if hit:
                idx, pr, tot = hit
                P *= 1.0 - tot
                P[idx] += pr

        ids = [self.index.get(t) for t in context]
        cache = Counter(i for t, i in zip(context[:-1], ids[:-1]) if i is not None and t not in CLOSED_MAP)
        ind = Counter()
        if context and WORD_RE.match(context[-1]):
            last = context[-1]
            last2 = context[-2] if len(context) >= 2 else None
            for p in range(len(context) - 1):
                nxt = ids[p + 1]
                if context[p] == last and nxt is not None:
                    both = last2 is not None and p > 0 and context[p - 1] == last2
                    ind[nxt] += IND_BIGRAM_WEIGHT if both else 1.0
        n_cached = sum(cache.values())
        lam_c = LAMBDA_CACHE * n_cached / (n_cached + CACHE_HALF)
        lam_i = LAMBDA_IND if ind else 0.0
        if ind and context[-1] in CLOSED_MAP:
            det = set(CLOSED_MAP[context[-1]]) & {"DET", "POSS"}
            lam_i *= IND_DET_FACTOR if det else IND_FUNCTION_FACTOR
        if lam_c or lam_i:
            P *= 1.0 - lam_c - lam_i
            for lam, counter in ((lam_c, cache), (lam_i, ind)):
                if lam:
                    idx = np.fromiter(counter.keys(), dtype=np.int64)
                    cnt = np.fromiter(counter.values(), dtype=np.float64)
                    P[idx] += lam * cnt / cnt.sum()
        return P
