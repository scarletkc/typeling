//! `regsvr32` support: the COM class, the input profile under English (United States) and the TSF categories.
//! All of it is per machine, so registering needs an elevated prompt.

use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, MAX_PATH, WIN32_ERROR};
use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance};
use windows::Win32::System::LibraryLoader::GetModuleFileNameW;
use windows::Win32::System::Registry::{
    HKEY, HKEY_LOCAL_MACHINE, KEY_WRITE, REG_OPTION_NON_VOLATILE, REG_SZ, RegCloseKey, RegCreateKeyExW, RegDeleteTreeW,
    RegSetValueExW,
};
use windows::Win32::UI::Input::KeyboardAndMouse::HKL;
use windows::Win32::UI::TextServices::{
    CLSID_TF_CategoryMgr, CLSID_TF_InputProcessorProfiles, GUID_TFCAT_TIP_KEYBOARD, GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT,
    GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT, ITfCategoryMgr, ITfInputProcessorProfileMgr,
};
use windows_core::{Error, GUID, HSTRING, PCWSTR, Result, w};

use crate::{CLSID_TEXT_SERVICE, GUID_PROFILE, bar::module};

const LANG_EN_US: u16 = 0x0409;
const DISPLAY_NAME: &str = "Typeling";
const CATEGORIES: [GUID; 3] =
    [GUID_TFCAT_TIP_KEYBOARD, GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT, GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT];

fn class_key() -> HSTRING {
    HSTRING::from(format!(r"Software\Classes\CLSID\{{{CLSID_TEXT_SERVICE:?}}}"))
}

fn check(code: WIN32_ERROR) -> Result<()> {
    code.ok()
}

fn set_string(key: HKEY, name: PCWSTR, value: &HSTRING) -> Result<()> {
    let bytes: Vec<u8> = value.iter().chain(&[0u16]).flat_map(|c| c.to_le_bytes()).collect();
    check(unsafe { RegSetValueExW(key, name, None, REG_SZ, Some(&bytes)) })
}

fn module_path() -> Result<HSTRING> {
    let mut buf = vec![0u16; MAX_PATH as usize * 4];
    let len = unsafe { GetModuleFileNameW(Some(module()), &mut buf) } as usize;
    if len == 0 || len == buf.len() {
        return Err(Error::from_thread());
    }
    Ok(HSTRING::from_wide(&buf[..len]))
}

fn register_class() -> Result<()> {
    let path = module_path()?;
    let subkey = HSTRING::from(format!(r"{}\InprocServer32", class_key()));
    let mut key = HKEY::default();
    unsafe {
        check(RegCreateKeyExW(
            HKEY_LOCAL_MACHINE,
            &subkey,
            None,
            PCWSTR::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE,
            None,
            &mut key,
            None,
        ))?;
        let result = set_string(key, PCWSTR::null(), &path)
            .and_then(|()| set_string(key, w!("ThreadingModel"), &HSTRING::from("Apartment")));
        let _ = RegCloseKey(key);
        result
    }
}

pub(crate) fn register() -> Result<()> {
    register_class()?;
    let name: Vec<u16> = DISPLAY_NAME.encode_utf16().collect();
    unsafe {
        let profiles: ITfInputProcessorProfileMgr =
            CoCreateInstance(&CLSID_TF_InputProcessorProfiles, None, CLSCTX_INPROC_SERVER)?;
        profiles.RegisterProfile(
            &CLSID_TEXT_SERVICE,
            LANG_EN_US,
            &GUID_PROFILE,
            &name,
            &[],
            0,
            HKL::default(),
            0,
            true,
            0,
        )?;
        let categories: ITfCategoryMgr = CoCreateInstance(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER)?;
        for category in &CATEGORIES {
            categories.RegisterCategory(&CLSID_TEXT_SERVICE, category, &CLSID_TEXT_SERVICE)?;
        }
    }
    Ok(())
}

/// Removes as much as it can, so a partly finished registration can be cleaned up too.
pub(crate) fn unregister() -> Result<()> {
    let mut first_error = None;
    let mut note = |r: Result<()>| {
        if let Err(e) = r {
            first_error.get_or_insert(e);
        }
    };
    unsafe {
        match CoCreateInstance::<_, ITfCategoryMgr>(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER) {
            Ok(categories) => {
                for category in &CATEGORIES {
                    let _ = categories.UnregisterCategory(&CLSID_TEXT_SERVICE, category, &CLSID_TEXT_SERVICE);
                }
            }
            Err(e) => note(Err(e)),
        }
        match CoCreateInstance::<_, ITfInputProcessorProfileMgr>(
            &CLSID_TF_InputProcessorProfiles,
            None,
            CLSCTX_INPROC_SERVER,
        ) {
            Ok(profiles) => {
                let _ = profiles.UnregisterProfile(&CLSID_TEXT_SERVICE, LANG_EN_US, &GUID_PROFILE, 0);
            }
            Err(e) => note(Err(e)),
        }
        let deleted = RegDeleteTreeW(HKEY_LOCAL_MACHINE, &class_key());
        if deleted != ERROR_FILE_NOT_FOUND && deleted != ERROR_PATH_NOT_FOUND {
            note(check(deleted));
        }
    }
    match first_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
