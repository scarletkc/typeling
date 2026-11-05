//! Typeling text service: a Windows input method (TSF) that shows English word candidates next to the cursor.
//!
//! The DLL is loaded into every application the user types in, so it stays small: it reads the text before the
//! cursor, asks the engine process over a named pipe from a worker thread, and draws the candidate bar. Letters reach
//! the application untouched; the service takes a key only to accept or dismiss a visible bar.
#![cfg(windows)]

mod bar;
mod client;
mod logic;
mod register;
mod service;

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use windows::Win32::Foundation::{CLASS_E_CLASSNOTAVAILABLE, CLASS_E_NOAGGREGATION, E_FAIL, E_POINTER, S_FALSE};
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl};
use windows_core::{BOOL, GUID, HRESULT, IUnknown, Interface, Ref, Result, implement};

pub use bar::BAR_CLASS;
pub use client::{PIPE_NAME, requests_sent};

pub const CLSID_TEXT_SERVICE: GUID = GUID::from_u128(0xe3e49a7b_dd4f_40ad_9712_10c378b8283b);
pub const GUID_PROFILE: GUID = GUID::from_u128(0xf0756bcd_30f3_4894_9c9a_e7c7b2710719);
/// Preserved keys: Alt+1, Alt+2 and Alt+3 pick a candidate.
pub const GUID_PICK_KEYS: [GUID; 3] = [
    GUID::from_u128(0x6d9643b2_c3ca_4475_89b9_bf6dbd4c63e5),
    GUID::from_u128(0x16e2da23_6978_4f99_be1c_42e734f3ec02),
    GUID::from_u128(0x13a7d958_7c5c_43fa_b51c_8d53a613a515),
];

/// A panic must never cross into the host application: across `extern "system"` it would abort the process.
pub(crate) fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err(E_FAIL.into()))
}

#[implement(IClassFactory)]
struct ClassFactory;

impl IClassFactory_Impl for ClassFactory_Impl {
    fn CreateInstance(&self, outer: Ref<IUnknown>, iid: *const GUID, object: *mut *mut c_void) -> Result<()> {
        guard(|| {
            if object.is_null() || iid.is_null() {
                return Err(E_POINTER.into());
            }
            unsafe { *object = std::ptr::null_mut() };
            if outer.is_some() {
                return Err(CLASS_E_NOAGGREGATION.into());
            }
            let service: IUnknown = service::Service::new().into();
            unsafe { service.query(iid, object).ok() }
        })
    }

    fn LockServer(&self, _lock: BOOL) -> Result<()> {
        Ok(())
    }
}

/// # Safety
/// COM entry point: the pointers must be valid, as COM guarantees.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllGetClassObject(clsid: *const GUID, iid: *const GUID, object: *mut *mut c_void) -> HRESULT {
    if clsid.is_null() || iid.is_null() || object.is_null() {
        return E_POINTER;
    }
    unsafe {
        *object = std::ptr::null_mut();
        if *clsid != CLSID_TEXT_SERVICE {
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        let factory: IClassFactory = ClassFactory.into();
        factory.query(iid, object)
    }
}

/// The worker thread and the window procedure live in this DLL, so it stays loaded until the process exits.
#[unsafe(no_mangle)]
pub extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

#[unsafe(no_mangle)]
pub extern "system" fn DllRegisterServer() -> HRESULT {
    match guard(register::register) {
        Ok(()) => HRESULT(0),
        Err(e) => {
            let _ = guard(register::unregister);
            e.code()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn DllUnregisterServer() -> HRESULT {
    match guard(register::unregister) {
        Ok(()) => HRESULT(0),
        Err(e) => e.code(),
    }
}
