//! The text service: follows the focused document, reads the text before the cursor and applies accepted candidates.
//!
//! Everything here runs on the host application's UI thread, and calls back into TSF can re-enter it, so state lives
//! in small cells whose borrows never span a call into TSF.

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::rc::Rc;

use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::Security::{GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation};
use windows::Win32::System::Com::CoTaskMemFree;
use windows::Win32::System::SystemInformation::GetTickCount64;
use windows::Win32::System::Threading::{GetCurrentProcess, GetCurrentThreadId, OpenProcessToken};
use windows::Win32::System::Variant::{VT_UNKNOWN, VariantClear};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetFocus, GetKeyState, VIRTUAL_KEY, VK_CONTROL, VK_ESCAPE, VK_MENU, VK_SHIFT, VK_TAB,
};
use windows::Win32::UI::TextServices::{
    GUID_PROP_INPUTSCOPE, IS_ALPHANUMERIC_PIN, IS_ALPHANUMERIC_PIN_SET, IS_NUMERIC_PASSWORD, IS_NUMERIC_PIN, IS_PASSWORD,
    IS_PRIVATE, ITfContext, ITfDocumentMgr, ITfEditRecord, ITfEditSession, ITfEditSession_Impl, ITfInputScope,
    ITfKeyEventSink, ITfKeyEventSink_Impl, ITfKeystrokeMgr, ITfRange, ITfSource,
    ITfTextEditSink, ITfTextEditSink_Impl, ITfTextInputProcessor_Impl, ITfTextInputProcessorEx,
    ITfTextInputProcessorEx_Impl, ITfThreadFocusSink, ITfThreadFocusSink_Impl, ITfThreadMgr, ITfThreadMgrEventSink,
    ITfThreadMgrEventSink_Impl, InputScope, TF_AE_NONE, TF_ANCHOR_END, TF_CONTEXT_EDIT_CONTEXT_FLAGS,
    TF_DEFAULT_SELECTION, TF_ES_ASYNCDONTCARE, TF_ES_READ, TF_ES_READWRITE, TF_INVALID_COOKIE,
    TF_MOD_ALT, TF_PRESERVEDKEY, TF_SD_READONLY, TF_SELECTION, TF_SELECTIONSTYLE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DefWindowProcW, GUITHREADINFO, GWLP_USERDATA, GetClassNameW, GetGUIThreadInfo, GetWindowLongPtrW, GetWindowLongW,
    GWL_STYLE, KillTimer, MA_NOACTIVATE, SetTimer, WM_LBUTTONUP, WM_MOUSEACTIVATE, WM_PAINT, WM_TIMER,
};
use windows_core::{BOOL, GUID, IUnknownImpl, Interface, Ref, Result, implement};

use crate::bar::Bar;
use crate::client::{Job, WM_REPLY, Worker};
use crate::logic::{EDIT_WINDOW_MS, MAX_CONTEXT, Reply, is_edit_key, is_modifier, wants_prediction};
use crate::{GUID_PICK_KEYS, guard};

const TIMER_REFRESH: usize = 1;
const TIMER_POSITION: usize = 2;
/// Some applications report text changes late or not at all, so a typed key also schedules a read.
const REFRESH_DELAY_MS: u32 = 40;
/// A browser may not have laid out the new text yet when the reply arrives.
const POSITION_DELAY_MS: u32 = 30;
const POSITION_RETRIES: u32 = 3;
const ES_PASSWORD: i32 = 0x20;
const PRIVATE_SCOPES: [InputScope; 6] =
    [IS_PASSWORD, IS_NUMERIC_PASSWORD, IS_NUMERIC_PIN, IS_ALPHANUMERIC_PIN, IS_ALPHANUMERIC_PIN_SET, IS_PRIVATE];

fn key_down(vk: VIRTUAL_KEY) -> bool {
    unsafe { GetKeyState(vk.0 as i32) < 0 }
}

/// Text typed in an elevated window must not cross into the engine, which runs without elevation.
fn is_elevated() -> bool {
    unsafe {
        let mut token = HANDLE::default();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token).is_err() {
            return true;
        }
        let mut elevation = TOKEN_ELEVATION::default();
        let mut size = 0;
        let known = GetTokenInformation(
            token,
            TokenElevation,
            Some(&mut elevation as *mut _ as *mut _),
            size_of::<TOKEN_ELEVATION>() as u32,
            &mut size,
        )
        .is_ok();
        let _ = CloseHandle(token);
        !known || elevation.TokenIsElevated != 0
    }
}

// Reading and changing the document. Each function needs an edit cookie (`ec`) from an edit session.

fn selection(context: &ITfContext, ec: u32) -> Option<ITfRange> {
    let mut selection = [TF_SELECTION::default()];
    let mut fetched = 0;
    unsafe {
        context.GetSelection(ec, TF_DEFAULT_SELECTION, &mut selection, &mut fetched).ok()?;
        if fetched == 0 {
            return None;
        }
        ManuallyDrop::take(&mut selection[0].range)
    }
}

fn set_selection(context: &ITfContext, ec: u32, range: &ITfRange) -> Result<()> {
    let mut selection = TF_SELECTION {
        range: ManuallyDrop::new(Some(range.clone())),
        style: TF_SELECTIONSTYLE { ase: TF_AE_NONE, fInterimChar: false.into() },
    };
    unsafe {
        let result = context.SetSelection(ec, std::slice::from_ref(&selection));
        ManuallyDrop::drop(&mut selection.range);
        result
    }
}

fn text_before(range: &ITfRange, ec: u32, max: i32) -> Result<Vec<u16>> {
    let mut buffer = vec![0u16; max as usize];
    let mut fetched = 0;
    let mut moved = 0;
    unsafe {
        let span = range.Clone()?;
        span.ShiftStart(ec, -max, &mut moved, std::ptr::null())?;
        span.GetText(ec, 0, &mut buffer, &mut fetched)?;
    }
    buffer.truncate(fetched as usize);
    Ok(buffer)
}

fn scope_is_private(context: &ITfContext, ec: u32, range: &ITfRange) -> bool {
    unsafe {
        let Ok(property) = context.GetAppProperty(&GUID_PROP_INPUTSCOPE) else { return false };
        let Ok(mut value) = property.GetValue(ec, range) else { return false };
        let mut private = false;
        if value.Anonymous.Anonymous.vt == VT_UNKNOWN
            && let Some(Ok(scope)) =
                value.Anonymous.Anonymous.Anonymous.punkVal.as_ref().map(|unknown| unknown.cast::<ITfInputScope>())
        {
            let mut scopes = std::ptr::null_mut();
            let mut count = 0;
            if scope.GetInputScopes(&mut scopes, &mut count).is_ok() && !scopes.is_null() {
                private =
                    std::slice::from_raw_parts(scopes, count as usize).iter().any(|s| PRIVATE_SCOPES.contains(s));
                CoTaskMemFree(Some(scopes as *const _));
            }
        }
        let _ = VariantClear(&mut value);
        private
    }
}

/// Classic Edit controls mark password fields with a window style rather than an input scope.
fn focus_is_password_edit() -> bool {
    unsafe {
        let focus = GetFocus();
        if focus.is_invalid() {
            return false;
        }
        let mut class = [0u16; 16];
        let len = GetClassNameW(focus, &mut class) as usize;
        String::from_utf16_lossy(&class[..len]).eq_ignore_ascii_case("edit")
            && GetWindowLongW(focus, GWL_STYLE) & ES_PASSWORD != 0
    }
}

/// The text before the cursor, or None when this field must get no predictions.
fn predictable_text(context: &ITfContext, ec: u32) -> Option<Vec<u16>> {
    unsafe {
        if context.GetStatus().ok()?.dwDynamicFlags & TF_SD_READONLY != 0 {
            return None;
        }
        let range = selection(context, ec)?;
        if !range.IsEmpty(ec).ok()?.as_bool() {
            return None;
        }
        if scope_is_private(context, ec, &range) || focus_is_password_edit() {
            return None;
        }
        text_before(&range, ec, MAX_CONTEXT).ok()
    }
}

fn is_blank(rect: &RECT) -> bool {
    rect.left == 0 && rect.top == 0 && rect.right == 0 && rect.bottom == 0
}

fn caret_rect(context: &ITfContext, ec: u32) -> Option<RECT> {
    unsafe {
        let range = selection(context, ec)?;
        let view = context.GetActiveView().ok()?;
        let mut rect = RECT::default();
        let mut clipped = BOOL::default();
        view.GetTextExt(ec, &range, &mut rect, &mut clipped).ok()?;
        (!is_blank(&rect)).then_some(rect)
    }
}

/// The classic system caret, for applications that cannot report text positions.
fn system_caret_rect() -> Option<RECT> {
    unsafe {
        let mut info = GUITHREADINFO { cbSize: size_of::<GUITHREADINFO>() as u32, ..Default::default() };
        GetGUIThreadInfo(GetCurrentThreadId(), &mut info).ok()?;
        if info.hwndCaret.is_invalid() || is_blank(&info.rcCaret) {
            return None;
        }
        let mut top_left = POINT { x: info.rcCaret.left, y: info.rcCaret.top };
        if !ClientToScreen(info.hwndCaret, &mut top_left).as_bool() {
            return None;
        }
        Some(RECT {
            left: top_left.x,
            top: top_left.y,
            right: top_left.x + (info.rcCaret.right - info.rcCaret.left),
            bottom: top_left.y + (info.rcCaret.bottom - info.rcCaret.top),
        })
    }
}

/// Put `word` in place of the partly typed word, unless the document has moved on since the suggestion was made.
fn apply_candidate(context: &ITfContext, ec: u32, word: &[u16], partial: &[u16]) -> Result<()> {
    let Some(cursor) = selection(context, ec) else { return Ok(()) };
    unsafe {
        if !cursor.IsEmpty(ec)?.as_bool() {
            return Ok(());
        }
        if !partial.is_empty() && text_before(&cursor, ec, partial.len() as i32)? != partial {
            return Ok(());
        }
        // Touch only what changes: most candidates just add letters to the typed ones.
        let target = cursor.Clone()?;
        let text = match word.strip_prefix(partial) {
            Some(tail) => tail,
            None => {
                let mut moved = 0;
                target.ShiftStart(ec, -(partial.len() as i32), &mut moved, std::ptr::null())?;
                word
            }
        };
        target.SetText(ec, 0, text)?;
        target.Collapse(ec, TF_ANCHOR_END)?;
        set_selection(context, ec, &target)
    }
}

type SessionWork = Box<dyn FnOnce(u32)>;

#[implement(ITfEditSession)]
struct Session {
    work: RefCell<Option<SessionWork>>,
}

impl ITfEditSession_Impl for Session_Impl {
    fn DoEditSession(&self, ec: u32) -> Result<()> {
        guard(|| {
            let work = self.work.borrow_mut().take();
            if let Some(work) = work {
                work(ec);
            }
            Ok(())
        })
    }
}

fn request_session(
    context: &ITfContext,
    client: u32,
    access: TF_CONTEXT_EDIT_CONTEXT_FLAGS,
    work: impl FnOnce(u32) + 'static,
) {
    let session: ITfEditSession = Session { work: RefCell::new(Some(Box::new(work))) }.into();
    // Runs at once when the document is free, otherwise as soon as it is.
    let _ = unsafe { context.RequestEditSession(client, &session, access | TF_ES_ASYNCDONTCARE) };
}

struct Suggestion {
    words: Vec<String>,
    partial: Vec<u16>,
}

enum KeyAction {
    Accept,
    Dismiss,
}

/// State shared by the COM object, the edit sessions and the window procedure.
pub(crate) struct Shared {
    client: Cell<u32>,
    thread_mgr: RefCell<Option<ITfThreadMgr>>,
    context: RefCell<Option<ITfContext>>,
    context_cookie: Cell<u32>,
    thread_cookies: RefCell<Vec<u32>>,
    bar: Bar,
    worker: RefCell<Option<Worker>>,
    /// Identifies the newest request; replies to older ones are dropped.
    generation: Cell<u64>,
    queried: RefCell<Vec<u16>>,
    suggestion: RefCell<Option<Suggestion>>,
    last_edit_key: Cell<u64>,
    eaten_key: Cell<u32>,
    position_retries: Cell<u32>,
}

impl Shared {
    fn new() -> Self {
        Shared {
            client: Cell::new(0),
            thread_mgr: RefCell::new(None),
            context: RefCell::new(None),
            context_cookie: Cell::new(TF_INVALID_COOKIE),
            thread_cookies: RefCell::new(Vec::new()),
            bar: Bar::default(),
            worker: RefCell::new(None),
            generation: Cell::new(0),
            queried: RefCell::new(Vec::new()),
            suggestion: RefCell::new(None),
            last_edit_key: Cell::new(0),
            eaten_key: Cell::new(0),
            position_retries: Cell::new(0),
        }
    }

    fn current_context(&self) -> Option<ITfContext> {
        self.context.borrow().clone()
    }

    fn recently_typed(&self) -> bool {
        let last = self.last_edit_key.get();
        last != 0 && unsafe { GetTickCount64() }.saturating_sub(last) <= EDIT_WINDOW_MS
    }

    /// Forget the current suggestion and any request still under way, and hide the bar.
    fn clear(&self) {
        self.generation.set(self.generation.get() + 1);
        self.queried.borrow_mut().clear();
        *self.suggestion.borrow_mut() = None;
        self.bar.hide();
        if let Some(window) = self.bar.hwnd() {
            unsafe {
                let _ = KillTimer(Some(window), TIMER_POSITION);
            }
        }
    }

    /// The document may have changed. Ask for candidates if the user is typing, otherwise dismiss the bar.
    fn on_document_change(&self, context: &ITfContext, ec: u32) {
        if !self.recently_typed() {
            self.clear();
            return;
        }
        let text = match predictable_text(context, ec) {
            Some(text) if wants_prediction(&text) => text,
            _ => {
                self.clear();
                return;
            }
        };
        if *self.queried.borrow() == text {
            return;
        }
        self.queried.borrow_mut().clone_from(&text);
        let generation = self.generation.get() + 1;
        self.generation.set(generation);
        let mut worker = self.worker.borrow_mut();
        if worker.is_none() {
            *worker = self.bar.hwnd().and_then(Worker::spawn);
        }
        if let Some(worker) = worker.as_ref() {
            worker.send(Job { generation, text });
        }
    }

    fn on_reply(self: &Rc<Self>, generation: u64, reply: Reply) {
        if generation != self.generation.get() {
            return;
        }
        if reply.words.is_empty() {
            *self.suggestion.borrow_mut() = None;
            self.bar.hide();
            return;
        }
        let partial = {
            let queried = self.queried.borrow();
            queried[queried.len().saturating_sub(reply.partial)..].to_vec()
        };
        *self.suggestion.borrow_mut() = Some(Suggestion { words: reply.words, partial });
        self.position_retries.set(0);
        self.show_bar();
    }

    fn show_bar(self: &Rc<Self>) {
        let Some(context) = self.current_context() else {
            self.clear();
            return;
        };
        let this = self.clone();
        let document = context.clone();
        request_session(&context, self.client.get(), TF_ES_READ, move |ec| this.place_bar(&document, ec));
    }

    fn place_bar(&self, context: &ITfContext, ec: u32) {
        let words = match self.suggestion.borrow().as_ref() {
            Some(suggestion) => suggestion.words.clone(),
            None => return,
        };
        if let Some(rect) = caret_rect(context, ec).or_else(system_caret_rect) {
            self.bar.show(rect, &words);
        } else if self.position_retries.get() < POSITION_RETRIES {
            self.position_retries.set(self.position_retries.get() + 1);
            if let Some(window) = self.bar.hwnd() {
                unsafe { SetTimer(Some(window), TIMER_POSITION, POSITION_DELAY_MS, None) };
            }
        } else {
            self.bar.hide(); // nowhere to put it; Tab stays Tab
        }
    }

    fn accept(&self, index: usize, context: &ITfContext) -> bool {
        let chosen = self.suggestion.borrow().as_ref().and_then(|s| {
            s.words.get(index).map(|word| (word.encode_utf16().collect::<Vec<u16>>(), s.partial.clone()))
        });
        let Some((word, partial)) = chosen else { return false };
        // The bar stays away until the next typed key, rather than offering to extend the accepted word.
        self.last_edit_key.set(0);
        self.clear();
        let document = context.clone();
        request_session(context, self.client.get(), TF_ES_READWRITE, move |ec| {
            let _ = apply_candidate(&document, ec, &word, &partial);
        });
        true
    }

    fn key_action(&self, vk: u32) -> Option<KeyAction> {
        if !self.bar.visible() || self.suggestion.borrow().is_none() {
            return None;
        }
        if vk == VK_ESCAPE.0 as u32 {
            return Some(KeyAction::Dismiss);
        }
        let modified = key_down(VK_SHIFT) || key_down(VK_CONTROL) || key_down(VK_MENU);
        (vk == VK_TAB.0 as u32 && !modified).then_some(KeyAction::Accept)
    }

    /// A key the service does not take: note whether it is typing or the user has moved on.
    fn note_key(&self, vk: u32) {
        if is_modifier(vk) {
            return;
        }
        if is_edit_key(vk, key_down(VK_CONTROL), key_down(VK_MENU)) {
            self.last_edit_key.set(unsafe { GetTickCount64() });
            if let Some(window) = self.bar.hwnd() {
                unsafe { SetTimer(Some(window), TIMER_REFRESH, REFRESH_DELAY_MS, None) };
            }
        } else {
            self.last_edit_key.set(0);
            self.clear();
        }
    }

    fn on_timer(self: &Rc<Self>, id: usize) {
        let Some(window) = self.bar.hwnd() else { return };
        unsafe {
            let _ = KillTimer(Some(window), id);
        }
        match id {
            TIMER_REFRESH if self.recently_typed() => {
                if let Some(context) = self.current_context() {
                    let this = self.clone();
                    let document = context.clone();
                    request_session(&context, self.client.get(), TF_ES_READ, move |ec| {
                        this.on_document_change(&document, ec);
                    });
                }
            }
            TIMER_POSITION => self.show_bar(),
            _ => {}
        }
    }
}

unsafe extern "system" fn bar_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let handled = guard(|| {
        unsafe {
            if message == WM_MOUSEACTIVATE {
                return Ok(Some(LRESULT(MA_NOACTIVATE as isize)));
            }
            let pointer = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const Shared;
            if pointer.is_null() {
                if message == WM_REPLY {
                    drop(Box::from_raw(lparam.0 as *mut Reply));
                    return Ok(Some(LRESULT(0)));
                }
                return Ok(None);
            }
            // The Service owns a reference for as long as the pointer is set; take another for this call.
            Rc::increment_strong_count(pointer);
            let shared = Rc::from_raw(pointer);
            Ok(match message {
                WM_REPLY => {
                    let reply = Box::from_raw(lparam.0 as *mut Reply);
                    shared.on_reply(wparam.0 as u64, *reply);
                    Some(LRESULT(0))
                }
                WM_TIMER => {
                    shared.on_timer(wparam.0);
                    Some(LRESULT(0))
                }
                WM_PAINT => {
                    shared.bar.paint();
                    Some(LRESULT(0))
                }
                WM_LBUTTONUP => {
                    let x = (lparam.0 & 0xFFFF) as i16 as i32;
                    if let (Some(index), Some(context)) = (shared.bar.hit(x), shared.current_context()) {
                        shared.accept(index, &context);
                    }
                    Some(LRESULT(0))
                }
                _ => None,
            })
        }
    });
    match handled {
        Ok(Some(result)) => result,
        _ => unsafe { DefWindowProcW(hwnd, message, wparam, lparam) },
    }
}

#[implement(ITfTextInputProcessorEx, ITfThreadMgrEventSink, ITfThreadFocusSink, ITfTextEditSink, ITfKeyEventSink)]
pub(crate) struct Service {
    shared: Rc<Shared>,
}

impl Service {
    pub fn new() -> Self {
        Service { shared: Rc::new(Shared::new()) }
    }
}

impl Service_Impl {
    fn start(&self, thread_mgr: &ITfThreadMgr, client: u32) -> Result<()> {
        let shared = &self.shared;
        shared.client.set(client);
        *shared.thread_mgr.borrow_mut() = Some(thread_mgr.clone());
        shared.bar.create(Some(bar_proc), Rc::as_ptr(shared) as isize)?;

        let object = self.to_object();
        unsafe {
            let source: ITfSource = thread_mgr.cast()?;
            let events: ITfThreadMgrEventSink = object.to_interface();
            shared.thread_cookies.borrow_mut().push(source.AdviseSink(&ITfThreadMgrEventSink::IID, &events)?);
            let focus: ITfThreadFocusSink = object.to_interface();
            shared.thread_cookies.borrow_mut().push(source.AdviseSink(&ITfThreadFocusSink::IID, &focus)?);

            let keystrokes: ITfKeystrokeMgr = thread_mgr.cast()?;
            let keys: ITfKeyEventSink = object.to_interface();
            keystrokes.AdviseKeyEventSink(client, &keys, true)?;
            for (i, guid) in GUID_PICK_KEYS.iter().enumerate() {
                let key = TF_PRESERVEDKEY { uVKey: b'1' as u32 + i as u32, uModifiers: TF_MOD_ALT };
                let description: Vec<u16> = format!("Pick candidate {}", i + 1).encode_utf16().collect();
                keystrokes.PreserveKey(client, guid, &key, &description)?;
            }
            let focused = thread_mgr.GetFocus().ok();
            self.follow(focused.as_ref());
        }
        Ok(())
    }

    fn stop(&self) {
        let shared = &self.shared;
        self.unfollow();
        shared.clear();
        let thread_mgr = shared.thread_mgr.borrow_mut().take();
        if let Some(thread_mgr) = thread_mgr {
            unsafe {
                if let Ok(keystrokes) = thread_mgr.cast::<ITfKeystrokeMgr>() {
                    for (i, guid) in GUID_PICK_KEYS.iter().enumerate() {
                        let key = TF_PRESERVEDKEY { uVKey: b'1' as u32 + i as u32, uModifiers: TF_MOD_ALT };
                        let _ = keystrokes.UnpreserveKey(guid, &key);
                    }
                    let _ = keystrokes.UnadviseKeyEventSink(shared.client.get());
                }
                if let Ok(source) = thread_mgr.cast::<ITfSource>() {
                    for cookie in shared.thread_cookies.borrow_mut().drain(..) {
                        let _ = source.UnadviseSink(cookie);
                    }
                }
            }
        }
        *shared.worker.borrow_mut() = None;
        shared.bar.destroy();
    }

    /// Watch the top context of the focused document.
    fn follow(&self, document: Option<&ITfDocumentMgr>) {
        self.unfollow();
        let Some(context) = document.and_then(|d| unsafe { d.GetTop() }.ok()) else { return };
        let Ok(source) = context.cast::<ITfSource>() else { return };
        let sink: ITfTextEditSink = self.to_object().to_interface();
        if let Ok(cookie) = unsafe { source.AdviseSink(&ITfTextEditSink::IID, &sink) } {
            self.shared.context_cookie.set(cookie);
            *self.shared.context.borrow_mut() = Some(context);
        }
    }

    fn unfollow(&self) {
        let shared = &self.shared;
        shared.last_edit_key.set(0);
        shared.clear();
        let context = shared.context.borrow_mut().take();
        let cookie = shared.context_cookie.replace(TF_INVALID_COOKIE);
        if cookie != TF_INVALID_COOKIE
            && let Some(source) = context.and_then(|c| c.cast::<ITfSource>().ok())
        {
            let _ = unsafe { source.UnadviseSink(cookie) };
        }
    }

    fn refollow(&self) {
        let thread_mgr = self.shared.thread_mgr.borrow().clone();
        if let Some(thread_mgr) = thread_mgr {
            let focused = unsafe { thread_mgr.GetFocus() }.ok();
            self.follow(focused.as_ref());
        }
    }

    fn context_for_key(&self, pic: &Ref<ITfContext>) -> Option<ITfContext> {
        pic.as_ref().cloned().or_else(|| self.shared.current_context())
    }
}

impl ITfTextInputProcessor_Impl for Service_Impl {
    fn Activate(&self, thread_mgr: Ref<ITfThreadMgr>, client: u32) -> Result<()> {
        ITfTextInputProcessorEx_Impl::ActivateEx(self, thread_mgr, client, 0)
    }

    fn Deactivate(&self) -> Result<()> {
        guard(|| {
            self.stop();
            Ok(())
        })
    }
}

impl ITfTextInputProcessorEx_Impl for Service_Impl {
    fn ActivateEx(&self, thread_mgr: Ref<ITfThreadMgr>, client: u32, _flags: u32) -> Result<()> {
        guard(|| {
            if is_elevated() {
                return Ok(()); // stay inert: letters still reach the application
            }
            let result = self.start(thread_mgr.ok()?, client);
            if result.is_err() {
                self.stop();
            }
            result
        })
    }
}

impl ITfThreadMgrEventSink_Impl for Service_Impl {
    fn OnInitDocumentMgr(&self, _document: Ref<ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    fn OnUninitDocumentMgr(&self, _document: Ref<ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    fn OnSetFocus(&self, focus: Ref<ITfDocumentMgr>, _previous: Ref<ITfDocumentMgr>) -> Result<()> {
        guard(|| {
            self.follow(focus.as_ref());
            Ok(())
        })
    }

    fn OnPushContext(&self, _context: Ref<ITfContext>) -> Result<()> {
        guard(|| {
            self.refollow();
            Ok(())
        })
    }

    fn OnPopContext(&self, _context: Ref<ITfContext>) -> Result<()> {
        guard(|| {
            self.refollow();
            Ok(())
        })
    }
}

impl ITfThreadFocusSink_Impl for Service_Impl {
    fn OnSetThreadFocus(&self) -> Result<()> {
        Ok(())
    }

    fn OnKillThreadFocus(&self) -> Result<()> {
        guard(|| {
            self.shared.last_edit_key.set(0);
            self.shared.clear();
            Ok(())
        })
    }
}

impl ITfTextEditSink_Impl for Service_Impl {
    fn OnEndEdit(&self, context: Ref<ITfContext>, ec: u32, _record: Ref<ITfEditRecord>) -> Result<()> {
        guard(|| {
            self.shared.on_document_change(context.ok()?, ec);
            Ok(())
        })
    }
}

impl ITfKeyEventSink_Impl for Service_Impl {
    fn OnSetFocus(&self, foreground: BOOL) -> Result<()> {
        guard(|| {
            if !foreground.as_bool() {
                self.shared.last_edit_key.set(0);
                self.shared.clear();
            }
            Ok(())
        })
    }

    fn OnTestKeyDown(&self, _context: Ref<ITfContext>, wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        guard(|| {
            let vk = wparam.0 as u32;
            let eat = self.shared.key_action(vk).is_some();
            if !eat {
                self.shared.note_key(vk);
            }
            Ok(eat.into())
        })
    }

    fn OnKeyDown(&self, context: Ref<ITfContext>, wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        guard(|| {
            let vk = wparam.0 as u32;
            let shared = &self.shared;
            let eaten = match shared.key_action(vk) {
                Some(KeyAction::Accept) => self.context_for_key(&context).is_some_and(|c| shared.accept(0, &c)),
                Some(KeyAction::Dismiss) => {
                    shared.last_edit_key.set(0);
                    shared.clear();
                    true
                }
                None => {
                    shared.note_key(vk);
                    false
                }
            };
            if eaten {
                shared.eaten_key.set(vk);
            }
            Ok(eaten.into())
        })
    }

    fn OnTestKeyUp(&self, _context: Ref<ITfContext>, wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        Ok((self.shared.eaten_key.get() == wparam.0 as u32).into())
    }

    fn OnKeyUp(&self, _context: Ref<ITfContext>, wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        let eaten = self.shared.eaten_key.get() == wparam.0 as u32;
        if eaten {
            self.shared.eaten_key.set(0);
        }
        Ok(eaten.into())
    }

    fn OnPreservedKey(&self, context: Ref<ITfContext>, key: *const GUID) -> Result<BOOL> {
        guard(|| {
            let shared = &self.shared;
            if key.is_null() || !shared.bar.visible() {
                return Ok(false.into());
            }
            let wanted = unsafe { *key };
            let Some(index) = GUID_PICK_KEYS.iter().position(|guid| *guid == wanted) else {
                return Ok(false.into());
            };
            Ok(self.context_for_key(&context).is_some_and(|c| shared.accept(index, &c)).into())
        })
    }
}
