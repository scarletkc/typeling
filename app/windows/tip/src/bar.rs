//! The candidate bar: a small window next to the cursor that never takes the keyboard focus.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;

use windows::Win32::Foundation::{COLORREF, ERROR_SUCCESS, HMODULE, HWND, POINT, RECT, SIZE};
use windows::Win32::Graphics::Dwm::{DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND, DwmSetWindowAttribute};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, BitBlt, CLEARTYPE_QUALITY, CLIP_DEFAULT_PRECIS, CreateCompatibleBitmap, CreateCompatibleDC, CreateFontW,
    CreatePen, CreateSolidBrush, DEFAULT_CHARSET, DeleteDC, DeleteObject, EndPaint, FF_DONTCARE, FW_NORMAL, FillRect,
    GetDC, GetMonitorInfoW, GetTextExtentPoint32W, HDC, HFONT, InvalidateRect, MONITOR_DEFAULTTONEAREST, MONITORINFO,
    MonitorFromPoint, OUT_DEFAULT_PRECIS, PAINTSTRUCT, PS_SOLID, ReleaseDC, RoundRect, SRCCOPY, SelectObject, SetBkMode,
    SetTextColor, TRANSPARENT, TextOutW,
};
use windows::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleHandleExW,
};
use windows::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_DWORD, RegGetValueW};
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::WindowsAndMessaging::{
    CS_DROPSHADOW, CreateWindowExW, DestroyWindow, GWLP_USERDATA, GetClientRect, HWND_TOPMOST, IsWindowVisible,
    RegisterClassExW, SW_HIDE, SWP_NOACTIVATE, SWP_SHOWWINDOW, SetWindowLongPtrW, SetWindowPos, SetWindowTextW,
    ShowWindow, WNDCLASSEXW, WNDPROC, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP,
};
use windows_core::{HSTRING, PCWSTR, w};

pub const BAR_CLASS: &str = "TypelingCandidateBar";

// Sizes at 96 DPI.
const HEIGHT: i32 = 36;
const FONT_PX: i32 = 15;
const OUTER_PAD: i32 = 4;
const ITEM_PAD: i32 = 10;
const ITEM_GAP: i32 = 2;
const LABEL_GAP: i32 = 6;
const CARET_GAP: i32 = 4;

struct Item {
    label: Vec<u16>,
    word: Vec<u16>,
    label_width: i32,
    x: i32,
    width: i32,
}

struct Palette {
    background: COLORREF,
    text: COLORREF,
    label: COLORREF,
    highlight: COLORREF,
    accent: COLORREF,
}

const fn rgb(r: u8, g: u8, b: u8) -> COLORREF {
    COLORREF(r as u32 | (g as u32) << 8 | (b as u32) << 16)
}

const LIGHT: Palette = Palette {
    background: rgb(0xF9, 0xF9, 0xF9),
    text: rgb(0x1A, 0x1A, 0x1A),
    label: rgb(0x70, 0x70, 0x70),
    highlight: rgb(0xE9, 0xE9, 0xE9),
    accent: rgb(0x00, 0x67, 0xC0),
};
const DARK: Palette = Palette {
    background: rgb(0x2C, 0x2C, 0x2C),
    text: rgb(0xF2, 0xF2, 0xF2),
    label: rgb(0xA8, 0xA8, 0xA8),
    highlight: rgb(0x40, 0x40, 0x40),
    accent: rgb(0x4C, 0xC2, 0xFF),
};

fn palette() -> &'static Palette {
    let mut light = 1u32;
    let mut size = size_of::<u32>() as u32;
    let found = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            w!(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"),
            w!("AppsUseLightTheme"),
            RRF_RT_REG_DWORD,
            None,
            Some(&mut light as *mut u32 as *mut c_void),
            Some(&mut size),
        )
    };
    if found == ERROR_SUCCESS && light == 0 { &DARK } else { &LIGHT }
}

/// This DLL's handle; found from an address so it also works when the crate is linked into a test.
pub(crate) fn module() -> HMODULE {
    let mut handle = HMODULE::default();
    let address = module as *const () as *const u16;
    let _ = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCWSTR(address),
            &mut handle,
        )
    };
    handle
}

fn scale(value: i32, dpi: u32) -> i32 {
    value * dpi as i32 / 96
}

fn make_font(dpi: u32) -> HFONT {
    unsafe {
        CreateFontW(
            -scale(FONT_PX, dpi),
            0,
            0,
            0,
            FW_NORMAL.0 as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS,
            CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY,
            FF_DONTCARE.0 as u32,
            w!("Segoe UI"),
        )
    }
}

fn text_width(hdc: HDC, text: &[u16]) -> i32 {
    let mut size = SIZE::default();
    let _ = unsafe { GetTextExtentPoint32W(hdc, text, &mut size) };
    size.cx
}

fn key_label(index: usize) -> String {
    if index == 0 { "Tab".to_owned() } else { format!("Alt+{}", index + 1) }
}

#[derive(Default)]
pub(crate) struct Bar {
    window: Cell<Option<isize>>,
    items: RefCell<Vec<Item>>,
}

impl Bar {
    pub fn hwnd(&self) -> Option<HWND> {
        self.window.get().map(|w| HWND(w as *mut c_void))
    }

    pub fn create(&self, proc: WNDPROC, user_data: isize) -> windows_core::Result<HWND> {
        unsafe {
            let instance = module().into();
            let class = HSTRING::from(BAR_CLASS);
            let info = WNDCLASSEXW {
                cbSize: size_of::<WNDCLASSEXW>() as u32,
                style: CS_DROPSHADOW,
                lpfnWndProc: proc,
                hInstance: instance,
                lpszClassName: PCWSTR(class.as_ptr()),
                ..Default::default()
            };
            RegisterClassExW(&info); // fails harmlessly when another UI thread has registered it
            let hwnd = CreateWindowExW(
                WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                &class,
                PCWSTR::null(),
                WS_POPUP,
                0,
                0,
                0,
                0,
                None,
                None,
                Some(instance),
                None,
            )?;
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, user_data);
            let corner = DWMWCP_ROUND;
            let _ = DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                &corner as *const _ as *const c_void,
                size_of_val(&corner) as u32,
            );
            self.window.set(Some(hwnd.0 as isize));
            Ok(hwnd)
        }
    }

    pub fn destroy(&self) {
        if let Some(hwnd) = self.hwnd() {
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                let _ = DestroyWindow(hwnd);
            }
        }
        self.window.set(None);
        self.items.borrow_mut().clear();
    }

    pub fn visible(&self) -> bool {
        self.hwnd().is_some_and(|hwnd| unsafe { IsWindowVisible(hwnd).as_bool() })
    }

    pub fn hide(&self) {
        if let Some(hwnd) = self.hwnd() {
            let _ = unsafe { ShowWindow(hwnd, SW_HIDE) };
        }
    }

    /// Lay out the words and show the bar below `caret` (screen coordinates), or above it near the screen's bottom.
    pub fn show(&self, caret: RECT, words: &[String]) {
        let Some(hwnd) = self.hwnd() else { return };
        unsafe {
            let dpi = GetDpiForWindow(hwnd).max(96);
            let hdc = GetDC(Some(hwnd));
            let font = make_font(dpi);
            let old = SelectObject(hdc, font.into());
            let mut x = scale(OUTER_PAD, dpi);
            let mut items = Vec::with_capacity(words.len());
            for (i, word) in words.iter().enumerate() {
                let label: Vec<u16> = key_label(i).encode_utf16().collect();
                let word: Vec<u16> = word.encode_utf16().collect();
                let label_width = text_width(hdc, &label);
                let width = scale(ITEM_PAD, dpi) * 2 + label_width + scale(LABEL_GAP, dpi) + text_width(hdc, &word);
                items.push(Item { label, word, label_width, x, width });
                x += width + scale(ITEM_GAP, dpi);
            }
            SelectObject(hdc, old);
            let _ = DeleteObject(font.into());
            ReleaseDC(Some(hwnd), hdc);
            let width = x - scale(ITEM_GAP, dpi) + scale(OUTER_PAD, dpi);
            let height = scale(HEIGHT, dpi);
            *self.items.borrow_mut() = items;

            let (left, top) = place(caret, width, height, scale(CARET_GAP, dpi));
            let _ = SetWindowTextW(hwnd, &HSTRING::from(words.join(" ")));
            let _ = SetWindowPos(hwnd, Some(HWND_TOPMOST), left, top, width, height, SWP_NOACTIVATE | SWP_SHOWWINDOW);
            let _ = InvalidateRect(Some(hwnd), None, true);
        }
    }

    /// Index of the candidate at client coordinate `x`.
    pub fn hit(&self, x: i32) -> Option<usize> {
        self.items.borrow().iter().position(|item| x >= item.x && x < item.x + item.width)
    }

    pub fn paint(&self) {
        let Some(hwnd) = self.hwnd() else { return };
        unsafe {
            let colors = palette();
            let mut ps = PAINTSTRUCT::default();
            let target = BeginPaint(hwnd, &mut ps);
            let mut client = RECT::default();
            let _ = GetClientRect(hwnd, &mut client);
            let (width, height) = (client.right, client.bottom);
            let dpi = GetDpiForWindow(hwnd).max(96);

            // Draw off screen, so typing fast does not make the bar flicker.
            let hdc = CreateCompatibleDC(Some(target));
            let bitmap = CreateCompatibleBitmap(target, width, height);
            let old_bitmap = SelectObject(hdc, bitmap.into());
            let background = CreateSolidBrush(colors.background);
            FillRect(hdc, &client, background);
            let _ = DeleteObject(background.into());

            let font = make_font(dpi);
            let old_font = SelectObject(hdc, font.into());
            SetBkMode(hdc, TRANSPARENT);
            let mut metric = SIZE::default();
            let _ = GetTextExtentPoint32W(hdc, &[b'x' as u16], &mut metric);
            let text_y = (height - metric.cy) / 2;
            let inset = scale(OUTER_PAD, dpi);

            for (i, item) in self.items.borrow().iter().enumerate() {
                if i == 0 {
                    // The first candidate is what Tab accepts.
                    let radius = scale(6, dpi);
                    let brush = CreateSolidBrush(colors.highlight);
                    let pen = CreatePen(PS_SOLID, 1, colors.highlight);
                    let old_brush = SelectObject(hdc, brush.into());
                    let old_pen = SelectObject(hdc, pen.into());
                    let _ = RoundRect(hdc, item.x, inset, item.x + item.width, height - inset, radius, radius);
                    SelectObject(hdc, old_brush);
                    SelectObject(hdc, old_pen);
                    let _ = DeleteObject(brush.into());
                    let _ = DeleteObject(pen.into());

                    let accent = CreateSolidBrush(colors.accent);
                    let bar_height = scale(16, dpi);
                    let top = (height - bar_height) / 2;
                    let left = item.x + scale(2, dpi);
                    let mark = RECT { left, top, right: left + scale(3, dpi), bottom: top + bar_height };
                    FillRect(hdc, &mark, accent);
                    let _ = DeleteObject(accent.into());
                }
                let label_x = item.x + scale(ITEM_PAD, dpi);
                SetTextColor(hdc, colors.label);
                let _ = TextOutW(hdc, label_x, text_y, &item.label);
                SetTextColor(hdc, colors.text);
                let _ = TextOutW(hdc, label_x + item.label_width + scale(LABEL_GAP, dpi), text_y, &item.word);
            }

            SelectObject(hdc, old_font);
            let _ = DeleteObject(font.into());
            let _ = BitBlt(target, 0, 0, width, height, Some(hdc), 0, 0, SRCCOPY);
            SelectObject(hdc, old_bitmap);
            let _ = DeleteObject(bitmap.into());
            let _ = DeleteDC(hdc);
            let _ = EndPaint(hwnd, &ps);
        }
    }
}

/// Keep the bar on the monitor that holds the caret.
fn place(caret: RECT, width: i32, height: i32, gap: i32) -> (i32, i32) {
    unsafe {
        let monitor = MonitorFromPoint(POINT { x: caret.left, y: caret.bottom }, MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO { cbSize: size_of::<MONITORINFO>() as u32, ..Default::default() };
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return (caret.left, caret.bottom + gap);
        }
        let area = info.rcWork;
        let left = caret.left.min(area.right - width).max(area.left);
        let below = caret.bottom + gap;
        let top = if below + height <= area.bottom { below } else { (caret.top - gap - height).max(area.top) };
        (left, top)
    }
}
