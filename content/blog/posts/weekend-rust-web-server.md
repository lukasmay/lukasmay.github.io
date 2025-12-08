+++
title = "Weekend Project: Building a Rust Web Server"
date = 2025-12-07T15:00:00-05:00
draft = false
categories = ["systems"]
tags = ["weekend-project", "rust"]
ShowToc = true
ShowReadingTime = true
ShowBreadCrumbs = true
ShowPostNavLinks = true
ShowCodeCopyButtons = true
+++

This weekend I decided to dive into Rust by building a simple web server from scratch. I've been curious about Rust's memory safety guarantees and wanted to see how it feels to work with in practice.

## The Goal

Build a basic HTTP server that can:
- Handle GET requests
- Serve static files
- Parse URLs and route requests
- Do it all without any external web frameworks

## Getting Started

First, I set up a new Rust project:

```bash
cargo new rust-web-server
cd rust-web-server
```

The plan was to use only Rust's standard library to really understand what's happening under the hood.

## Building the HTTP Server

### Basic TCP Server

I started with a simple TCP server that listens for connections:

```rust
use std::net::{TcpListener, TcpStream};
use std::io::prelude::*;
use std::thread;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    println!("Server running on http://127.0.0.1:7878");

    for stream in listener.incoming() {
        let stream = stream.unwrap();
        thread::spawn(|| {
            handle_connection(stream);
        });
    }
}

fn handle_connection(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    stream.read(&mut buffer).unwrap();
    
    // Parse the request here...
}
```

### HTTP Request Parsing

The trickiest part was parsing HTTP requests manually. I had to handle:
- Request line parsing (method, path, HTTP version)
- Header parsing
- Body handling for POST requests

```rust
fn parse_request(request: &str) -> (String, String) {
    let lines: Vec<&str> = request.split('\n').collect();
    let request_line = lines[0];
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    
    if parts.len() >= 2 {
        (parts[0].to_string(), parts[1].to_string())
    } else {
        ("GET".to_string(), "/".to_string())
    }
}
```

### File Serving

For static file serving, I implemented a simple file reader with MIME type detection:

```rust
fn serve_file(path: &str) -> (String, Vec<u8>) {
    let full_path = format!("./static{}", path);
    
    match std::fs::read(&full_path) {
        Ok(contents) => {
            let content_type = match path.split('.').last() {
                Some("html") => "text/html",
                Some("css") => "text/css",
                Some("js") => "application/javascript",
                Some("json") => "application/json",
                _ => "text/plain",
            };
            (content_type.to_string(), contents)
        }
        Err(_) => {
            ("text/html".to_string(), b"<h1>404 Not Found</h1>".to_vec())
        }
    }
}
```

## Challenges & Solutions

### Memory Management
Coming from languages with garbage collection, Rust's ownership system took some getting used to. The borrow checker caught several bugs that would have been runtime errors in other languages.

**Challenge**: Sharing data between threads
**Solution**: Used `Arc` (Atomically Reference Counted) smart pointers for shared state

### HTTP Protocol Details
I underestimated the complexity of HTTP parsing. Things like:
- Handling different line endings (`\r\n` vs `\n`)
- Case-insensitive header names
- Content-Length calculation

**Solution**: Spent time reading the HTTP/1.1 RFC and implementing proper parsing step by step.

### Performance Considerations
The thread-per-connection model doesn't scale well, but it was simple to implement for this learning exercise.

## What I Learned

### Rust-Specific Insights
1. **Ownership System**: Once you understand it, it actually makes code more reliable
2. **Pattern Matching**: The `match` expressions are incredibly powerful
3. **Error Handling**: `Result<T, E>` types force you to think about error cases
4. **Zero-Cost Abstractions**: High-level code compiles to efficient machine code

### HTTP Protocol Deep Dive
1. **Request Parsing**: HTTP is more complex than it appears on the surface
2. **Status Codes**: Understanding when to use different response codes
3. **Headers**: The importance of proper Content-Type and Content-Length headers
4. **Connection Handling**: Keep-alive vs. close connection strategies

### Systems Programming
1. **TCP Sockets**: How data flows at the network level
2. **Threading**: When and why to use concurrent processing
3. **File I/O**: Efficient file reading and MIME type detection

## Performance Results

With basic load testing using `wrk`:
```bash
wrk -t12 -c400 -d30s http://127.0.0.1:7878/
```

Results:
- **Requests/sec**: ~15,000
- **Latency avg**: 25ms
- **Memory usage**: ~8MB

Not bad for a learning project!

## Next Steps

This was just the beginning. Some improvements I want to explore:

1. **Async I/O**: Replace threads with async/await for better scalability
2. **HTTP/2 Support**: Implement the newer protocol features
3. **Middleware System**: Add a plugin architecture for request processing
4. **Configuration**: Make the server configurable via config files
5. **Testing**: Add comprehensive unit and integration tests

## The Code

The complete code is available on my GitHub. It's about 200 lines of Rust and demonstrates the core concepts without external dependencies.

## Final Thoughts

Building this web server was an excellent way to learn Rust. The language's emphasis on safety and performance really shines in systems programming scenarios like this.

The ownership system, while initially frustrating, caught several bugs before runtime. The type system is expressive enough to model complex scenarios while remaining readable.

Would I use this in production? Absolutely not. But as a learning exercise, it was perfect for understanding both Rust and HTTP at a deeper level.

Next weekend: maybe I'll tackle implementing a basic database engine in Rust? 🤔