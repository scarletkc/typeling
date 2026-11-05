//! End-to-end test without installing anything: a real TSF thread manager and the real engine, with a small in-memory
//! text store playing the application. The service is created through DllGetClassObject and activated directly, so
//! no registry entries are needed. Needs `python` on PATH.
#![cfg(windows)]

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::time::{Duration, Instant};

use typeling_tip::{BAR_CLASS, CLSID_TEXT_SERVICE, DllGetClassObject, GUID_PICK_KEYS, PIPE_NAME, requests_sent};
use windows::Win32::Foundation::{E_NOTIMPL, E_UNEXPECTED, HWND, LPARAM, POINT, RECT, WPARAM};
use windows::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemAlloc, FORMATETC,
    IClassFactory, IDataObject,
};
use windows::Win32::System::Variant::VT_UNKNOWN;
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, FindWindowW, GetWindowTextW, IsWindowVisible, MSG, PM_REMOVE, PeekMessageW, TranslateMessage,
};
use windows_core::{BOOL, BSTR, GUID, HRESULT, HSTRING, IUnknown, Interface, PCWSTR, PWSTR, Ref, Result, implement};

const VK_BACK: usize = 0x08;
const VK_TAB: usize = 0x09;
const VK_RETURN: usize = 0x0D;
const VK_ESCAPE: usize = 0x1B;
const VK_SPACE: usize = 0x20;
const VK_LEFT: usize = 0x25;
const VK_OEM_7: usize = 0xDE;

#[derive(Default)]
struct Doc {
    text: RefCell<Vec<u16>>,
    selection: Cell<(i32, i32)>,
    sink: RefCell<Option<ITextStoreACPSink>>,
    lock: Cell<u32>,
    queued: Cell<u32>,
    password: Cell<bool>,
    scope_requested: Cell<bool>,
}

impl Doc {
    fn replace(&self, start: i32, end: i32, new: &[u16]) -> TS_TEXTCHANGE {
        let mut text = self.text.borrow_mut();
        text.splice(start as usize..end as usize, new.iter().copied());
        let new_end = start + new.len() as i32;
        self.selection.set((new_end, new_end));
        TS_TEXTCHANGE { acpStart: start, acpOldEnd: end, acpNewEnd: new_end }
    }

    fn string(&self) -> String {
        String::from_utf16_lossy(&self.text.borrow())
    }

    /// What an application does when the user's key has reached it.
    fn type_text(&self, s: &str) {
        let (start, end) = self.selection.get();
        let new: Vec<u16> = s.encode_utf16().collect();
        let change = self.replace(start, end, &new);
        self.notify(Some(change));
    }

    fn backspace(&self) {
        let (start, _) = self.selection.get();
        let change = self.replace(start - 1, start, &[]);
        self.notify(Some(change));
    }

    fn move_cursor(&self, position: i32) {
        self.selection.set((position, position));
        self.notify(None);
    }

    fn notify(&self, change: Option<TS_TEXTCHANGE>) {
        let sink = self.sink.borrow().clone().unwrap();
        unsafe {
            if let Some(change) = change {
                sink.OnTextChange(TS_ST_NONE, &change).unwrap();
            }
            sink.OnSelectionChange().unwrap();
        }
    }
}

#[implement(ITextStoreACP)]
struct Store {
    doc: Rc<Doc>,
}

impl ITextStoreACP_Impl for Store_Impl {
    fn AdviseSink(&self, _riid: *const GUID, punk: Ref<IUnknown>, _mask: u32) -> Result<()> {
        *self.doc.sink.borrow_mut() = Some(punk.ok()?.cast()?);
        Ok(())
    }

    fn UnadviseSink(&self, _punk: Ref<IUnknown>) -> Result<()> {
        *self.doc.sink.borrow_mut() = None;
        Ok(())
    }

    fn RequestLock(&self, flags: u32) -> Result<HRESULT> {
        let sink = self.doc.sink.borrow().clone().ok_or(windows_core::Error::from(E_UNEXPECTED))?;
        if self.doc.lock.get() != 0 {
            if flags & TS_LF_SYNC != 0 {
                return Ok(TS_E_SYNCHRONOUS);
            }
            self.doc.queued.set(self.doc.queued.get() | flags);
            return Ok(TS_S_ASYNC);
        }
        let mut grant = flags & !TS_LF_SYNC;
        let mut result = HRESULT(0);
        while grant != 0 {
            self.doc.lock.set(grant);
            let granted = unsafe { sink.OnLockGranted(TEXT_STORE_LOCK_FLAGS(grant)) };
            self.doc.lock.set(0);
            if let Err(e) = granted {
                result = e.code();
            }
            grant = self.doc.queued.replace(0) & !TS_LF_SYNC;
        }
        Ok(result)
    }

    fn GetStatus(&self) -> Result<TS_STATUS> {
        Ok(TS_STATUS { dwDynamicFlags: 0, dwStaticFlags: TS_SS_NOHIDDENTEXT })
    }

    fn QueryInsert(&self, start: i32, end: i32, _cch: u32, result_start: *mut i32, result_end: *mut i32) -> Result<()> {
        unsafe {
            *result_start = start;
            *result_end = end;
        }
        Ok(())
    }

    fn GetSelection(&self, _index: u32, _count: u32, selection: *mut TS_SELECTION_ACP, fetched: *mut u32) -> Result<()> {
        if self.doc.lock.get() == 0 {
            return Err(TS_E_NOLOCK.into());
        }
        let (start, end) = self.doc.selection.get();
        unsafe {
            *selection = TS_SELECTION_ACP {
                acpStart: start,
                acpEnd: end,
                style: TS_SELECTIONSTYLE { ase: TS_AE_END, fInterimChar: false.into() },
            };
            *fetched = 1;
        }
        Ok(())
    }

    fn SetSelection(&self, _count: u32, selection: *const TS_SELECTION_ACP) -> Result<()> {
        let selection = unsafe { *selection };
        self.doc.selection.set((selection.acpStart, selection.acpEnd));
        Ok(())
    }

    fn GetText(
        &self,
        start: i32,
        end: i32,
        plain: PWSTR,
        plain_req: u32,
        plain_ret: *mut u32,
        runs: *mut TS_RUNINFO,
        runs_req: u32,
        runs_ret: *mut u32,
        next: *mut i32,
    ) -> Result<()> {
        if self.doc.lock.get() == 0 {
            return Err(TS_E_NOLOCK.into());
        }
        let text = self.doc.text.borrow();
        let end = if end < 0 { text.len() as i32 } else { end.min(text.len() as i32) };
        let available = (end - start).max(0);
        // A caller can ask for the runs alone, to measure text without copying it.
        let wants_plain = plain_req > 0 && !plain.is_null();
        let count = if wants_plain { available.min(plain_req as i32) } else { available };
        unsafe {
            if wants_plain && count > 0 {
                std::ptr::copy_nonoverlapping(text[start as usize..].as_ptr(), plain.0, count as usize);
            }
            if !plain_ret.is_null() {
                *plain_ret = if wants_plain { count as u32 } else { 0 };
            }
            let mut run_count = 0;
            if runs_req > 0 && count > 0 && !runs.is_null() {
                *runs = TS_RUNINFO { uCount: count as u32, r#type: TS_RT_PLAIN };
                run_count = 1;
            }
            if !runs_ret.is_null() {
                *runs_ret = run_count;
            }
            if !next.is_null() {
                *next = start + count;
            }
        }
        Ok(())
    }

    fn SetText(&self, _flags: u32, start: i32, end: i32, text: &PCWSTR, cch: u32) -> Result<TS_TEXTCHANGE> {
        if self.doc.lock.get() & 2 == 0 {
            return Err(TS_E_NOLOCK.into());
        }
        let new = unsafe { std::slice::from_raw_parts(text.0, cch as usize) };
        Ok(self.doc.replace(start, end, new))
    }

    fn GetFormattedText(&self, _start: i32, _end: i32) -> Result<IDataObject> {
        Err(E_NOTIMPL.into())
    }

    fn GetEmbedded(&self, _position: i32, _service: *const GUID, _riid: *const GUID) -> Result<IUnknown> {
        Err(E_NOTIMPL.into())
    }

    fn QueryInsertEmbedded(&self, _service: *const GUID, _format: *const FORMATETC) -> Result<BOOL> {
        Ok(false.into())
    }

    fn InsertEmbedded(&self, _flags: u32, _start: i32, _end: i32, _data: Ref<IDataObject>) -> Result<TS_TEXTCHANGE> {
        Err(E_NOTIMPL.into())
    }

    fn InsertTextAtSelection(
        &self,
        flags: u32,
        text: &PCWSTR,
        cch: u32,
        result_start: *mut i32,
        result_end: *mut i32,
        change: *mut TS_TEXTCHANGE,
    ) -> Result<()> {
        if self.doc.lock.get() & 2 == 0 {
            return Err(TS_E_NOLOCK.into());
        }
        let (start, end) = self.doc.selection.get();
        unsafe {
            if flags & TS_IAS_QUERYONLY == 0 {
                let new = std::slice::from_raw_parts(text.0, cch as usize);
                let made = self.doc.replace(start, end, new);
                if !change.is_null() {
                    *change = made;
                }
                if !result_start.is_null() {
                    *result_start = made.acpStart;
                }
                if !result_end.is_null() {
                    *result_end = made.acpNewEnd;
                }
            } else {
                if !result_start.is_null() {
                    *result_start = start;
                }
                if !result_end.is_null() {
                    *result_end = start + cch as i32;
                }
            }
        }
        Ok(())
    }

    fn InsertEmbeddedAtSelection(
        &self,
        _flags: u32,
        _data: Ref<IDataObject>,
        _start: *mut i32,
        _end: *mut i32,
        _change: *mut TS_TEXTCHANGE,
    ) -> Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn RequestSupportedAttrs(&self, _flags: u32, count: u32, attrs: *const GUID) -> Result<()> {
        self.note_attrs(count, attrs);
        Ok(())
    }

    fn RequestAttrsAtPosition(&self, _position: i32, count: u32, attrs: *const GUID, _flags: u32) -> Result<()> {
        self.note_attrs(count, attrs);
        Ok(())
    }

    fn RequestAttrsTransitioningAtPosition(&self, _position: i32, _count: u32, _attrs: *const GUID, _flags: u32) -> Result<()> {
        Ok(())
    }

    fn FindNextAttrTransition(
        &self,
        start: i32,
        _halt: i32,
        _count: u32,
        _attrs: *const GUID,
        _flags: u32,
        next: *mut i32,
        found: *mut BOOL,
        offset: *mut i32,
    ) -> Result<()> {
        unsafe {
            *next = start;
            *found = false.into();
            *offset = 0;
        }
        Ok(())
    }

    fn RetrieveRequestedAttrs(&self, count: u32, values: *mut TS_ATTRVAL, fetched: *mut u32) -> Result<()> {
        unsafe {
            *fetched = 0;
            if count > 0 && self.doc.scope_requested.replace(false) && self.doc.password.get() {
                let scope: ITfInputScope = PasswordScope.into();
                let value = &mut *values;
                value.idAttr = GUID_PROP_INPUTSCOPE;
                value.dwOverlapId = 0;
                let variant = &mut value.varValue.Anonymous.Anonymous;
                variant.vt = VT_UNKNOWN;
                variant.Anonymous.punkVal = std::mem::ManuallyDrop::new(Some(scope.cast()?));
                *fetched = 1;
            }
        }
        Ok(())
    }

    fn GetEndACP(&self) -> Result<i32> {
        Ok(self.doc.text.borrow().len() as i32)
    }

    fn GetActiveView(&self) -> Result<u32> {
        Ok(1)
    }

    fn GetACPFromPoint(&self, _view: u32, _point: *const POINT, _flags: u32) -> Result<i32> {
        Err(E_NOTIMPL.into())
    }

    fn GetTextExt(&self, _view: u32, _start: i32, _end: i32, rect: *mut RECT, clipped: *mut BOOL) -> Result<()> {
        unsafe {
            *rect = RECT { left: 100, top: 100, right: 101, bottom: 120 };
            *clipped = false.into();
        }
        Ok(())
    }

    fn GetScreenExt(&self, _view: u32) -> Result<RECT> {
        Ok(RECT { left: 0, top: 0, right: 800, bottom: 600 })
    }

    fn GetWnd(&self, _view: u32) -> Result<HWND> {
        Ok(HWND::default())
    }
}

impl Store_Impl {
    fn note_attrs(&self, count: u32, attrs: *const GUID) {
        if count > 0 && !attrs.is_null() {
            let attrs = unsafe { std::slice::from_raw_parts(attrs, count as usize) };
            if attrs.contains(&GUID_PROP_INPUTSCOPE) {
                self.doc.scope_requested.set(true);
            }
        }
    }
}

#[implement(ITfInputScope)]
struct PasswordScope;

impl ITfInputScope_Impl for PasswordScope_Impl {
    fn GetInputScopes(&self, scopes: *mut *mut InputScope, count: *mut u32) -> Result<()> {
        unsafe {
            let memory = CoTaskMemAlloc(size_of::<InputScope>()) as *mut InputScope;
            *memory = IS_PASSWORD;
            *scopes = memory;
            *count = 1;
        }
        Ok(())
    }

    fn GetPhrase(&self, _phrases: *mut *mut BSTR, _count: *mut u32) -> Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn GetRegularExpression(&self) -> Result<BSTR> {
        Err(E_NOTIMPL.into())
    }

    fn GetSRGS(&self) -> Result<BSTR> {
        Err(E_NOTIMPL.into())
    }

    fn GetXML(&self) -> Result<BSTR> {
        Err(E_NOTIMPL.into())
    }
}

/// What the service registers for keys. TSF accepts key sinks only from installed, activated input methods, so the
/// test keeps them here and delivers keys itself, in the order TSF does.
#[derive(Default)]
struct KeyRouter {
    sink: RefCell<Option<ITfKeyEventSink>>,
    preserved: RefCell<Vec<(GUID, u32, u32)>>,
}

/// The real thread manager, except for the key registrations (see KeyRouter).
#[implement(ITfThreadMgr, ITfSource, ITfKeystrokeMgr)]
struct ThreadMgrProxy {
    real: ITfThreadMgr,
    router: Rc<KeyRouter>,
}

impl ITfThreadMgr_Impl for ThreadMgrProxy_Impl {
    fn Activate(&self) -> Result<u32> {
        Err(E_NOTIMPL.into())
    }

    fn Deactivate(&self) -> Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn CreateDocumentMgr(&self) -> Result<ITfDocumentMgr> {
        unsafe { self.real.CreateDocumentMgr() }
    }

    fn EnumDocumentMgrs(&self) -> Result<IEnumTfDocumentMgrs> {
        Err(E_NOTIMPL.into())
    }

    fn GetFocus(&self) -> Result<ITfDocumentMgr> {
        unsafe { self.real.GetFocus() }
    }

    fn SetFocus(&self, focus: Ref<ITfDocumentMgr>) -> Result<()> {
        unsafe { self.real.SetFocus(focus.as_ref()) }
    }

    fn AssociateFocus(&self, _hwnd: HWND, _new: Ref<ITfDocumentMgr>) -> Result<ITfDocumentMgr> {
        Err(E_NOTIMPL.into())
    }

    fn IsThreadFocus(&self) -> Result<BOOL> {
        Err(E_NOTIMPL.into())
    }

    fn GetFunctionProvider(&self, _clsid: *const GUID) -> Result<ITfFunctionProvider> {
        Err(E_NOTIMPL.into())
    }

    fn EnumFunctionProviders(&self) -> Result<IEnumTfFunctionProviders> {
        Err(E_NOTIMPL.into())
    }

    fn GetGlobalCompartment(&self) -> Result<ITfCompartmentMgr> {
        Err(E_NOTIMPL.into())
    }
}

impl ITfSource_Impl for ThreadMgrProxy_Impl {
    fn AdviseSink(&self, riid: *const GUID, sink: Ref<IUnknown>) -> Result<u32> {
        unsafe { self.real.cast::<ITfSource>()?.AdviseSink(riid, sink.ok()?) }
    }

    fn UnadviseSink(&self, cookie: u32) -> Result<()> {
        unsafe { self.real.cast::<ITfSource>()?.UnadviseSink(cookie) }
    }
}

impl ITfKeystrokeMgr_Impl for ThreadMgrProxy_Impl {
    fn AdviseKeyEventSink(&self, _client: u32, sink: Ref<ITfKeyEventSink>, foreground: BOOL) -> Result<()> {
        assert!(foreground.as_bool(), "the service must ask for keys as the foreground input method");
        *self.router.sink.borrow_mut() = Some(sink.ok()?.clone());
        Ok(())
    }

    fn UnadviseKeyEventSink(&self, _client: u32) -> Result<()> {
        *self.router.sink.borrow_mut() = None;
        Ok(())
    }

    fn GetForeground(&self) -> Result<GUID> {
        Err(E_NOTIMPL.into())
    }

    fn TestKeyDown(&self, _wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        Err(E_NOTIMPL.into())
    }

    fn TestKeyUp(&self, _wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        Err(E_NOTIMPL.into())
    }

    fn KeyDown(&self, _wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        Err(E_NOTIMPL.into())
    }

    fn KeyUp(&self, _wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        Err(E_NOTIMPL.into())
    }

    fn GetPreservedKey(&self, _context: Ref<ITfContext>, _key: *const TF_PRESERVEDKEY) -> Result<GUID> {
        Err(E_NOTIMPL.into())
    }

    fn IsPreservedKey(&self, _guid: *const GUID, _key: *const TF_PRESERVEDKEY) -> Result<BOOL> {
        Err(E_NOTIMPL.into())
    }

    fn PreserveKey(&self, _client: u32, guid: *const GUID, key: *const TF_PRESERVEDKEY, _desc: &PCWSTR, _len: u32) -> Result<()> {
        let (guid, key) = unsafe { (*guid, *key) };
        self.router.preserved.borrow_mut().push((guid, key.uVKey, key.uModifiers));
        Ok(())
    }

    fn UnpreserveKey(&self, guid: *const GUID, _key: *const TF_PRESERVEDKEY) -> Result<()> {
        let guid = unsafe { *guid };
        self.router.preserved.borrow_mut().retain(|(g, _, _)| *g != guid);
        Ok(())
    }

    fn SetPreservedKeyDescription(&self, _guid: *const GUID, _desc: &PCWSTR, _len: u32) -> Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn GetPreservedKeyDescription(&self, _guid: *const GUID) -> Result<BSTR> {
        Err(E_NOTIMPL.into())
    }

    fn SimulatePreservedKey(&self, _context: Ref<ITfContext>, _guid: *const GUID) -> Result<BOOL> {
        Err(E_NOTIMPL.into())
    }
}

struct Engine(Child);

impl Engine {
    fn start() -> Engine {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("engine.py");
        let child = Command::new("python").arg(script).stdout(Stdio::null()).spawn().expect("python on PATH");
        let deadline = Instant::now() + Duration::from_secs(20);
        while OpenOptions::new().read(true).write(true).open(PIPE_NAME).is_err() {
            assert!(Instant::now() < deadline, "the engine did not start");
            std::thread::sleep(Duration::from_millis(20));
        }
        Engine(child)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn pump_until(limit: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    loop {
        unsafe {
            let mut message = MSG::default();
            while PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
        if done() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn bar_window() -> Option<HWND> {
    unsafe { FindWindowW(&HSTRING::from(BAR_CLASS), PCWSTR::null()) }.ok()
}

fn bar_visible() -> bool {
    bar_window().is_some_and(|hwnd| unsafe { IsWindowVisible(hwnd).as_bool() })
}

fn bar_words() -> Vec<String> {
    let Some(hwnd) = bar_window() else { return Vec::new() };
    let mut buffer = [0u16; 256];
    let len = unsafe { GetWindowTextW(hwnd, &mut buffer) } as usize;
    String::from_utf16_lossy(&buffer[..len]).split(' ').map(str::to_owned).collect()
}

struct Session {
    doc: Rc<Doc>,
    router: Rc<KeyRouter>,
    context: ITfContext,
}

impl Session {
    fn sink(&self) -> ITfKeyEventSink {
        self.router.sink.borrow().clone().expect("the service registered for keys")
    }

    /// A key press in the order TSF delivers it; returns whether the service took it.
    fn key(&self, vk: usize) -> bool {
        let sink = self.sink();
        unsafe {
            let (w, l) = (WPARAM(vk), LPARAM(1));
            let tested = sink.OnTestKeyDown(&self.context, w, l).unwrap().as_bool();
            let eaten = sink.OnKeyDown(&self.context, w, l).unwrap().as_bool();
            assert_eq!(tested, eaten, "OnTestKeyDown and OnKeyDown disagree on key {vk:#x}");
            let up_tested = sink.OnTestKeyUp(&self.context, w, l).unwrap().as_bool();
            let up = sink.OnKeyUp(&self.context, w, l).unwrap().as_bool();
            assert_eq!((up_tested, up), (eaten, eaten), "the key up must follow the key down for key {vk:#x}");
            eaten
        }
    }

    fn type_keys(&self, text: &str) {
        for ch in text.chars() {
            let vk = match ch {
                ' ' => VK_SPACE,
                '\n' => VK_RETURN,
                '\u{2019}' | '\'' => VK_OEM_7,
                c => c.to_ascii_uppercase() as usize,
            };
            assert!(!self.key(vk), "the service took the typed key {ch:?}");
            self.doc.type_text(&ch.to_string());
        }
    }

    fn pick(&self, index: usize) -> bool {
        unsafe { self.sink().OnPreservedKey(&self.context, &GUID_PICK_KEYS[index]).unwrap().as_bool() }
    }

    fn wait_for_bar(&self) {
        assert!(pump_until(Duration::from_secs(3), bar_visible), "no bar for {:?}", self.doc.string());
    }

    fn settle(&self) {
        pump_until(Duration::from_millis(250), || false);
    }
}

#[test]
fn typing_session() {
    let _engine = Engine::start();
    unsafe {
        CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok().unwrap();

        let manager: ITfThreadMgrEx = CoCreateInstance(&CLSID_TF_ThreadMgr, None, CLSCTX_INPROC_SERVER).unwrap();
        let mut client = 0;
        manager.ActivateEx(&mut client, TF_TMAE_NOACTIVATETIP).unwrap();
        let thread_mgr: ITfThreadMgr = manager.cast().unwrap();

        let doc = Rc::new(Doc::default());
        let store: ITextStoreACP = Store { doc: doc.clone() }.into();
        let document = thread_mgr.CreateDocumentMgr().unwrap();
        let mut context = None;
        let mut edit_cookie = 0;
        document.CreateContext(client, 0, &store, &mut context, &mut edit_cookie).unwrap();
        let context = context.unwrap();
        document.Push(&context).unwrap();

        let mut factory: *mut c_void = std::ptr::null_mut();
        DllGetClassObject(&CLSID_TEXT_SERVICE, &IClassFactory::IID, &mut factory).ok().unwrap();
        let factory = IClassFactory::from_raw(factory);
        let service: ITfTextInputProcessorEx = factory.CreateInstance(None).unwrap();
        let _: ITfTextInputProcessor = service.cast().expect("TSF asks for the base interface");
        let router = Rc::new(KeyRouter::default());
        let proxy: ITfThreadMgr = ThreadMgrProxy { real: thread_mgr.clone(), router: router.clone() }.into();
        // A test thread has no window focus, so TSF would not deliver focus changes: focus the document before the
        // service starts, and deliver the later changes to the service directly.
        thread_mgr.SetFocus(&document).unwrap();
        service.ActivateEx(&proxy, client, 0).unwrap();
        let focus_events: ITfThreadMgrEventSink = service.cast().unwrap();

        let alt = TF_MOD_ALT;
        assert_eq!(
            *router.preserved.borrow(),
            vec![(GUID_PICK_KEYS[0], 0x31, alt), (GUID_PICK_KEYS[1], 0x32, alt), (GUID_PICK_KEYS[2], 0x33, alt)]
        );
        let s = Session { doc: doc.clone(), router: router.clone(), context };

        // Completing a word: the typed letters stay, Tab adds the rest.
        s.type_keys("Let me kn");
        s.wait_for_bar();
        assert_eq!(bar_words()[0], "know", "{:?}", bar_words());
        assert!(s.key(VK_TAB), "Tab must accept while the bar is visible");
        assert!(pump_until(Duration::from_secs(1), || doc.string() == "Let me know"), "{:?}", doc.string());
        assert_eq!(doc.selection.get(), (11, 11), "the cursor follows the accepted word");
        assert!(!bar_visible(), "accepting hides the bar");
        s.settle();
        assert!(!bar_visible(), "no bar until the next typed key");
        assert!(!s.key(VK_TAB), "with no bar, Tab belongs to the application");

        // Predicting the next word and dismissing the bar.
        s.type_keys(" ");
        s.wait_for_bar();
        assert_eq!(doc.string(), "Let me know ");
        assert!(s.key(VK_ESCAPE), "Esc must dismiss a visible bar");
        assert!(!bar_visible());
        assert!(!s.key(VK_ESCAPE), "with no bar, Esc belongs to the application");
        assert!(!s.key(VK_TAB));
        s.settle();
        assert!(!bar_visible(), "a dismissed bar stays away");

        // Picking another candidate with Alt+2.
        s.type_keys("wh");
        s.wait_for_bar();
        let words = bar_words();
        assert!(words.len() >= 2, "{words:?}");
        assert!(words.iter().all(|w| w.starts_with("wh")), "{words:?}");
        assert!(s.pick(1), "Alt+2 must pick while the bar is visible");
        let expected = format!("Let me know {}", words[1]);
        assert!(pump_until(Duration::from_secs(1), || doc.string() == expected), "{:?}", doc.string());
        assert!(!s.pick(0), "with no bar, Alt+1 belongs to the application");

        // Deleting a character asks again.
        s.type_keys(" tha");
        s.wait_for_bar();
        assert!(!s.key(VK_BACK));
        doc.backspace();
        assert!(pump_until(Duration::from_secs(3), || bar_visible() && bar_words().iter().all(|w| w.starts_with("th"))));

        // A candidate that does not extend the typed text replaces it (curly apostrophe -> straight).
        doc.type_text("\n");
        s.key(VK_RETURN);
        s.settle();
        assert!(!bar_visible(), "a blank line gets no bar");
        assert!(!s.key(VK_TAB), "on a blank line, Tab belongs to the application");
        s.type_keys("I don\u{2019}");
        s.wait_for_bar();
        assert!(bar_words().contains(&"don't".to_owned()), "{:?}", bar_words());
        assert!(s.key(VK_TAB));
        assert!(pump_until(Duration::from_secs(1), || doc.string().ends_with("I don't")), "{:?}", doc.string());
        let end = doc.text.borrow().len() as i32;
        assert_eq!(doc.selection.get(), (end, end));

        // Moving away dismisses the bar: with the arrow keys, and with no key at all (a mouse click).
        s.type_keys(" ");
        s.wait_for_bar();
        assert!(!s.key(VK_LEFT));
        assert!(!bar_visible(), "an arrow key dismisses the bar");
        s.type_keys("k");
        s.wait_for_bar();
        pump_until(Duration::from_millis(600), || false); // longer than the typing window
        doc.move_cursor(0);
        assert!(pump_until(Duration::from_secs(1), || !bar_visible()), "moving the cursor dismisses the bar");

        // Nothing typed in a password field reaches the engine.
        let sent = requests_sent();
        doc.password.set(true);
        doc.move_cursor(doc.text.borrow().len() as i32);
        s.type_keys(" secret");
        s.settle();
        assert_eq!(requests_sent(), sent, "a password field must send nothing to the engine");
        assert!(!bar_visible(), "a password field gets no bar");
        doc.password.set(false);
        s.type_keys(" a");
        s.wait_for_bar();

        // Losing the focus hides the bar, and it comes back with the focus and the next key.
        focus_events.OnSetFocus(None, &document).unwrap();
        assert!(!bar_visible(), "losing focus hides the bar");
        s.key(0x42);
        s.settle();
        assert!(!bar_visible(), "no document, no bar");
        focus_events.OnSetFocus(&document, None).unwrap();
        s.type_keys(" th");
        s.wait_for_bar();

        // Clean shutdown, with the bar still up.
        service.Deactivate().unwrap();
        assert!(bar_window().is_none(), "deactivating destroys the bar");
        assert!(router.sink.borrow().is_none() && router.preserved.borrow().is_empty(), "key registrations are undone");
        document.Pop(TF_POPF_ALL).unwrap();
        manager.Deactivate().unwrap();
    }
}
