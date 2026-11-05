//! Worker thread that talks to the engine, so a slow or missing engine can never block the application's UI thread.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{PostMessageW, WM_APP};

use crate::logic::{Reply, encode_request, parse_reply};

pub const PIPE_NAME: &str = r"\\.\pipe\LOCAL\typeling-engine";
/// Posted to the bar window: wParam is the generation, lParam a `Box<Reply>`.
pub(crate) const WM_REPLY: u32 = WM_APP + 1;

const MAX_REPLY: usize = 64 * 1024;
const ERROR_PIPE_BUSY: i32 = 231;

static REQUESTS: AtomicU64 = AtomicU64::new(0);

/// Requests this process has sent to the engine (tests check that private fields send none).
pub fn requests_sent() -> u64 {
    REQUESTS.load(Ordering::Relaxed)
}

pub(crate) struct Job {
    pub generation: u64,
    pub text: Vec<u16>,
}

pub(crate) struct Worker {
    jobs: Sender<Job>,
}

impl Worker {
    /// The thread ends when the Worker is dropped.
    pub fn spawn(window: HWND) -> Option<Worker> {
        let (jobs, inbox) = mpsc::channel();
        let window = window.0 as isize;
        std::thread::Builder::new()
            .name("typeling-client".into())
            .spawn(move || run(inbox, window))
            .ok()?;
        Some(Worker { jobs })
    }

    pub fn send(&self, job: Job) {
        let _ = self.jobs.send(job);
    }
}

fn run(inbox: Receiver<Job>, window: isize) {
    let mut pipe = None;
    while let Ok(mut job) = inbox.recv() {
        // Only the newest text matters when the user types faster than the engine answers.
        while let Ok(newer) = inbox.try_recv() {
            job = newer;
        }
        REQUESTS.fetch_add(1, Ordering::Relaxed);
        let request = encode_request(&job.text);
        let reply = exchange(&mut pipe, &request).unwrap_or(Reply { partial: 0, words: Vec::new() });
        post(window, job.generation, reply);
    }
}

fn post(window: isize, generation: u64, reply: Reply) {
    let raw = Box::into_raw(Box::new(reply));
    let posted = unsafe {
        PostMessageW(Some(HWND(window as *mut _)), WM_REPLY, WPARAM(generation as usize), LPARAM(raw as isize))
    };
    if posted.is_err() {
        drop(unsafe { Box::from_raw(raw) }); // the window is gone
    }
}

/// One request and reply; reconnects once, since the engine may have been restarted since the last request.
fn exchange(pipe: &mut Option<File>, request: &[u8]) -> Option<Reply> {
    for _ in 0..2 {
        if pipe.is_none() {
            *pipe = Some(connect().ok()?);
        }
        match roundtrip(pipe.as_mut()?, request) {
            Ok(body) => return parse_reply(&body),
            Err(_) => *pipe = None,
        }
    }
    None
}

fn connect() -> io::Result<File> {
    let open = || OpenOptions::new().read(true).write(true).open(PIPE_NAME);
    // The engine makes a new pipe instance right after each connection; a client arriving in between has to retry.
    for _ in 0..5 {
        match open() {
            Err(e) if e.raw_os_error() == Some(ERROR_PIPE_BUSY) => std::thread::sleep(Duration::from_millis(5)),
            result => return result,
        }
    }
    open()
}

fn roundtrip(pipe: &mut File, request: &[u8]) -> io::Result<Vec<u8>> {
    pipe.write_all(request)?;
    let mut head = [0u8; 4];
    pipe.read_exact(&mut head)?;
    let size = u32::from_le_bytes(head) as usize;
    if size > MAX_REPLY {
        return Err(io::ErrorKind::InvalidData.into());
    }
    let mut body = vec![0u8; size];
    pipe.read_exact(&mut body)?;
    Ok(body)
}
