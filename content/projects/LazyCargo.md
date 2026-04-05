+++
title = "LazyCargo"
date = 2025-12-07T22:25:58-05:00
draft = false
categories = ["Project"]
tags = ["reversing", "OT", "ICS", "malware"]
ShowToc = true
ShowReadingTime = true
ShowBreadCrumbs = true
ShowCodeCopyButtons = true
+++

## Introduction
This is meant to be an outline of what I found while reversing the **LazyCargo** malware sample. This malware sample is one part of the five pipedream/INCONTROLLER malware framework components discovered by several cybersecurity firms and government agencies. The LazyCargo malware is a Windows dropper for another module in the framework. I don't have access to any of the other components, so I wrote a payload to run the LazyCargo malware at the end of the analysis to verify my findings from the static analysis.

I am going to skip a lot of the background on the pipedream malware framework, as this post is focused on LazyCargo, but I would recommend looking into the other components, as pipedream is the seventh known ICS-specific malware ever discovered, and lots of cool things are happening in the other components. I will first explain at a high level what **LazyCargo** does and then perform a walkthrough with code snippets of what I found inside the module finishing off with how I got the malware to run on a windows system with a custom payload.
## What LazyCargo Does
**LazyCargo** is a Windows malware loader. It takes a payload and makes it run in ring 0 or in kernel space instead of user land. The way it goes about setting that up is with the mechanism  **Bring your own vulnerable driver (BYOVD)**. This is used to essentially create a vulnerability in the system by adding vulnerable software that operates in ring 0. Windows is smart enough to not let random code operate in ring 0 and for Windows to load a driver it has to be trusted or in other words signed. So the solution to this problem is find a signed driver that Windows will trust that is vulnerable so the operating system will load vulnerable code into ring 0. The next step in the attack chain is to then exploit the vulnerability in the newly loaded driver which in this case was **AsRockDrv.sys** which has this vulnerability: **CVE-2020-15368**. 

So the exploit chain so far is to first to load the **AsRockDrv.sys** driver. Then register the driver with the system which starts the driver as a system service. The final step in that chain is to exploit the vulnerability which allows the malware to load the payload into ring 0 giving the payload kernel level permissions. There are lots of smaller steps that **LazyCargo** takes to make each of these things happen but that is the general overview of what and how it operates.

## Static Analysis
### Initial Findings
To start off with I ran the binary through [string sifter](https://github.com/mandiant/stringsifter) which is a tool that uses floss to extract all the strings from a binary and then reorders them based on relevance to malware reverse engineering making the process of getting relevant strings much quicker. 

```txt
c:\asrock\work\asrocksdk_v0.0.69\asrrw\src\driver\src\objfre_win7_amd64\amd64\AsrDrv103.pdb

C:\Users\User1\Desktop\dev projects\SignSploit1\x64\Release\AsrDrv_exploit.pdb

Dhttp://crl.microsoft.com/pki/crl/products/MicrosoftCodeVerifRoot.crl0

2Terms of use at https://www.verisign.com/rpa (c)101.0,

c:\program files (x86)\microsoft visual studio\2017\enterprise\vc\tools\msvc\14.16.27023\include\xmemory0

bcrypt.dll

minkernel\crts\ucrt\src\appcrt\string\strnicmp.cpp

C:\AsRockDrv.sys

\REGISTRY\MACHINE\HARDWARE\RESOURCEMAP\System Resources\Physical Memory

\Registry\Machine\System\CurrentControlSet\Control\Class

ntoskrnl.exe
```

Here are some of the top strings that string sifter was able to find. Right off the bat it is clear that something is happening with the **AsRockDrv.sys** as it's listed in several different strings. Next their is repeated mentions of signing. `bcrypt.dll` is a Windows Cryptographic Primitives Library. `ntoskrnl.exe` is responsible for essential system services, including hardware virtualization, process management, memory management, and security reference monitoring. The registry keys also point to some sort of low level operations. Overall the main takeaways from a quick look at strings is that something is happening with the AsRockDrv.sys driver and what looks like components that will help load it.

The next tool that I run the binary through is [capa](https://github.com/mandiant/capa). This tool essentially tells you the capabilities of a binary (Highly recommend). For this analysis I will just show the default output and how helpful it is when starting to look at a binary file. 

> capa-output Image

**ATT&CK Tactics and MAEC Category** This section maps the binary's high-level execution flow to industry-standard threat frameworks, revealing its primary objective on the infected host.
- **It is a Launcher:** The MAEC category explicitly identifies this binary as a launcher, meaning its core purpose is to deliver and execute a secondary payload rather than acting as the final stage itself.
- **Service-Based Persistence:** It ensures it survives system reboots by establishing persistence through the creation and modification of a Windows Service.
- **Evasive Execution:** It attempts to fly under the radar by utilizing obfuscated files and executing its processes through system services.

**Malware Behavior Catalog (MBC)** The MBC breakdown highlights the specific technical behaviors the malware uses to interact with the system, modify files, and evade automated analysis.
- **Anti-Debugging:** The binary actively tries to detect if it is being analyzed by using timing and delay checks (specifically `GetTickCount`) to identify debuggers.
- **Cryptography:** It contains routines to encrypt and decrypt data, strongly suggesting the secondary payload or its configuration is encrypted within the file.
- **System Reconnaissance:** It actively queries the registry and searches for specific files and directories to understand its environment before deploying its payload.

**Detailed Capabilities** This detailed list exposes the exact, low-level functions compiled into the binary, giving us a direct roadmap of its internal mechanics.
- **Embedded Payload:** The scan confirms the presence of an embedded PE (Portable Executable) file, verifying exactly what the launcher is hiding.
- **BCrypt Usage:** It specifically relies on the `BCrypt` API to handle its data encryption and decryption routines.
- **Debug Info:** The binary is compiled in debug mode exposing their original local file paths, internal logging, and potentially the original source code structure. Which makes sense for why strings was as rewarding as it was.
- **Dynamic API Loading:** It links functions at runtime, a technique used to hide the true APIs it relies on from basic static analysis tools.

At this point I have enough information to help me in understanding what might be happening inside the binary to open it in ghidra. 

### Mapping Control Flow
The next step that I took was to figure out how everything that I had found up until this point was linked together. So I found the main function and started looking at the decompiled code to see what control logic was in place.

> NOTE: I have already gone through and renamed labels in ghidra.
#### 1. Ingesting the Malicious Payload
The malware expects an argument (the unsigned driver) upon execution. It immediately opens this file, calculates its size, allocates memory, and reads the malicious payload into a buffer in user-space.
```c
// Ensures an argument is passed
if (argc < 2) {
	printf("please set unsigned driver as argument to program!\n");
	goto LAB_END;
}

// Loads the target file (the unsigned driver)
HVar2 = OpenFile(*(LPCSTR *)(argv + 8), local_528, 0);
hFile = (HANDLE)(longlong)HVar2;

// Checks how big the file is and reserves RAM
file_size = GetFileSize(hFile, (LPDWORD)0x0);
vector_resize((longlong *)&UnsignedDriver_Vector, (ulonglong)file_size);

// Copies data from disk to reserved RAM
ReadFile(hFile, UnsignedDriver_Data, ...);
```

**Takeaway:** The malware doesn't contain the ultimate payload hardcoded within itself; it expects to load it dynamically from disk. This modular approach allows the attacker to swap out different malicious drivers without recompiling the loader.

#### 2. Dropping the Stepping Stone (The Vulnerable Driver)
Once the malicious payload is in memory, the malware drops a second file to disk: a known vulnerable AsRock driver (`AsRockDrv.sys`). It writes this file directly from a hardcoded byte array (`DAT_driver - bytes`) embedded within the executable.
```c
// This creates the vuln Driver from the code of the payload
_File = fopen("C:\\AsRockDrv.sys", "wb");
if (_File == (FILE *)0x0) { ... }

// Writes the vulnerable driver to the C: drive
fwrite(&DAT_driver-bytes, 0x8708, 1, _File);
fclose(_File);
```

**Takeaway:** This is the core of the BYOVD technique. Because `AsRockDrv.sys` is likely a legitimately signed driver (despite containing security flaws), Windows will allow it to be loaded into the kernel without triggering Driver Signature Enforcement (DSE) alerts.

#### 3. Establishing a Kernel Foothold
With the vulnerable driver dropped to disk, the malware uses the Windows Service Control Manager to register it as a system service, start it, and then open a handle to communicate with it directly.
```c
// Start of the driver registration with windows
hSCManager = OpenSCManagerW((LPCWSTR)0x0, (LPCWSTR)0x0, 2);
hService = CreateServiceA(hSCManager, "AsRockDrv", "AsRockDrv", 0xf01ff, 1, 2, 1,
					"C:\\AsRockDrv.sys", ...);

// Loads the driver into the kernel
BVar3 = StartServiceW(hService, 0, (LPCWSTR *)0x0);

// Opens communication line via device symlink
hDevice_AsRock = CreateFileW(L"\\??\\AsrDrv103", 0xc0000000, 7, ...);
```

**Takeaway:** The malware has successfully transitioned from user-space execution to having a functional, trusted communication pipeline (`\\??\\AsrDrv103`) directly into the Windows kernel.

#### 4. Payload Assembly and Exploitation
This is where the actual exploit occurs. The malware concatenates a shellcode header with the unsigned driver it read in Step 1. It then finds the physical RAM address of a target IOCTL handler and uses a custom wrapper function (`FUN_Driver - function`) to send an IOCTL code (`0x22e80c`) to the vulnerable AsRock driver.
```c
// Assembles the payload: Shellcode header + Unsigned Driver
memcpy(puStack_568, (undefined8 *)&DAT_140061b70, 0x7fb);
memcpy((undefined8 *)((longlong)puStack_568 + 0x7fb), UnsignedDriver_Data, ...);

// Locates target memory
phyisical-address = find-physical-ram-addr();

// Exploits the AsRock driver to write to kernel memory
local_548[0] = phyisical-address;
FUN_Driver-function(0x22e80c, (undefined4 *)local_548);

// Triggers the execution of the injected shellcode
(*DAT_14006c740)(hDevice_AsRock, 0, 0, 0);
```

**Takeaway:** The loader leverages a specific vulnerability (triggered via IOCTL `0x22e80c`) in the AsRock driver to achieve arbitrary kernel memory write capabilities. It uses this to overwrite memory and manually map/execute the malicious, unsigned driver—completely bypassing Windows OS protections.

### Exploit Specifics
With a deeper look at the underlying functions, the true mechanics of how LazyCargo weaponizes the AsRock driver become clear. It executes a highly precise sequence involving physical memory scanning, payload encryption, and low-level system calls to achieve Ring 0 execution.

#### 1. Hunting for the Target in Physical RAM
Before the malware can inject its payload, it needs to know exactly _where_ to write it. The `find - physical - ram - addr` function handles this by using the vulnerable driver as a memory scanner.
```c
// Scans memory using a read IOCTL (0x22e808)
Debug-check = FUN_Driver-function(0x22e808, (undefined4 *)&local_50);

// Compares the read memory against a hardcoded signature
if ((Debug-check == 0) && (Debug-check = memcmp(pvStack_90, &DAT_140062370, 0xa0), Debug-check == 0)) {
    GetTickCount();
    printf("\nfound map in %.3f sec physical address : %016I64x\n");
    goto FUN_no-debug-found;
}
```

**Takeaway:** The malware uses IOCTL `0x22e808` (which grants arbitrary physical memory read access) to iterate through RAM. It reads chunks of memory and uses `memcmp` to compare them against a specific 160-byte (`0xa0` hex) signature. This allows the malware to dynamically locate the exact physical address of the target kernel structure or function it intends to overwrite, bypassing memory randomization protections like ASLR.

#### 2. Evading Detection with Encrypted Payloads
When dispatching IOCTLs to the driver, LazyCargo doesn't send its data in the clear. The `FUN_Driver - function` acts as a specialized wrapper that utilizes the `BCrypt` API (Windows Cryptography Next Generation) to encrypt the payload parameters.
```c
// Initializes AES encryption via Windows CNG
NVar5 = BCryptOpenAlgorithmProvider(&local_b8, L"AES", (LPCWSTR)0x0, 0);

// ... (Key generation and buffer setup) ...

// Encrypts the IOCTL parameters/payload before sending
NVar5 = BCryptEncrypt(local_b0, pUStack_78, cbOutput-local_a8[0],
                      (void *)0x0, (PUCHAR)0x0, 0, pUStack_78, cbOutput,
                      local_c0, 1);
```

**Takeaway:** By utilizing AES to encrypt the IOCTL buffer, the malware achieves two critical objectives: it satisfies the specific cryptographic input requirements of this version of the AsRock driver, and it actively evades Endpoint Detection and Response (EDR) solutions that scan memory buffers for known plaintext shellcode patterns before they enter kernel space.

#### 3. Bypassing User-Mode Hooks (The Trigger)
To actually send the IOCTLs and trigger the execution, the malware actively avoids using the standard `DeviceIoControl` function found in `kernel32.dll`. Instead, it resolves the underlying NTAPI function directly.
```c
// Dynamically resolves the lowest-level user-mode API
hModule = GetModuleHandleA("ntdll.dll");
DAT_14006c740 = GetProcAddress(hModule, "NtDeviceIoControlFile");

// Later, the function pointer is used to send the IOCTL directly:
(*DAT_14006c740)(hDevice_AsRock, 0, 0, 0);
```

**Takeaway:** This is a classic user-mode hook evasion technique. Many security products monitor the higher-level `DeviceIoControl` API to catch malicious driver interactions. By dynamically resolving and calling `NtDeviceIoControlFile` straight from `ntdll.dll`, LazyCargo slips under those API hooks, ensuring its commands are handed directly to the kernel to trigger the final Ring 0 payload execution.

### Summary
From our static analysis, LazyCargo paints a clear picture: it is a purpose-built, highly evasive BYOVD loader. By dropping the vulnerable AsRock driver, scanning physical memory for its exact injection point, encrypting its IOCTL communications to blind EDRs, and bypassing standard API hooks via `NtDeviceIoControlFile`, it methodically paves a stealthy path straight to Ring 0.

But static analysis only gives us the blueprint; dynamic analysis is where we prove it works. Detonating this malware wasn't as simple as firing up a VM and watching it run. To truly verify my findings, I had to tackle the execution in three distinct phases. First, I needed to navigate the gauntlet of anti-debugging traps built into the binary just to get it to execute freely in my environment. Second, because LazyCargo acts as a reflective loader, I had to write and compile a bare-bones dummy driver to ensure the malware could actually load it into memory without immediately blue-screening the Windows kernel. Finally, once I achieved a stable load, I moved on to developing a more complex custom payload attempting the classic `calc.exe` pop to definitively prove that the injected code successfully executes with full kernel-level privileges.

## Dynamic Analysis
### Debugger Traps
While analyzing LazyCargo, I quickly realized that throwing this binary directly into a debugger wasn't going to be completely straightforward. The developers left behind side-effects of their build configuration and included time profiling that can make debugging quite annoying. I've broken down the two main tricks I found that hinder the analysis process.

#### 1. Time Profiling 
A technique flagged during my `capa` scan was the use of `GetTickCount`. I tracked this down to the `FUN_ai_find_physical_ram_addr` function, which is responsible for scanning memory. 
```c
  if (local_58 == '\0') {
    GetTickCount();
    if ((Debug_check == 0) && (Debug_check = memcmp(pvStack_90,&DAT_140062370,0xa0), Debug_check == 0)) {
	    GetTickCount();
        printf("\nfound map in %.3f sec physical address : %016I64x\n");
        goto FUN_no_debug_found;
    }
```

The malware records the system uptime immediately before and after its physical memory scan loop. While the primary purpose here appears to be calculating the elapsed time to print to the console, time delta checks like this are notoriously used to detect debuggers. If an analyst is manually stepping through this loop in a debugger, the time delta between the two `GetTickCount()` calls will be massive compared to a normal execution. This can inadvertently trigger anti-debugging behaviors if checked later, or simply alert the analyst that time is being monitored.

#### 2. Debug Build Artifacts & `INT 3` Traps
Interestingly, this malware sample was compiled as a Debug build. Because of this, it includes Microsoft Visual C++ runtime assertions (specifically `_CrtDbgReport` for `std::vector` out-of-bounds checks). 

```c
Debug_check = _CrtDbgReport(2, "c:\\program files (x86)\\microsoft visual studio\\2017\\enterprise\\vc\\tools\\msvc\\14.16.27023\\include\\vector",0x6c5,(char *)0x0,"%s");

if (Debug_check == 1) {
	var_int3_trigger = (code *)swi(3);
	(*var_int3_trigger)();
	return;
}
```

If you are debugging the malware and trigger one of these bounds checks, the CRT will pop an assertion dialog. If you click "Retry" (which returns `1`), the malware executes a software interrupt (`swi(3)`), which translates to an `INT 3` instruction. This acts as a hardcoded breakpoint. If you aren't expecting it or your debugger doesn't handle the exception properly, it breaks the execution flow entirely and makes dynamic analysis incredibly frustrating.

Another thing that you have to get around is that the binary expects a payload. While you can skip over many of these checks it becomes increasingly hard when LazyCargo is trying to load the payload and it doesn't find anything. This means to get the malware to run to completion I would need to give LazyCargo a payload that would not crash the system when loaded into the kernel. 

### Dummy Driver
The dummy driver is the first attempt at this as I was not sure what was going to be needed. I started out trying to compile a binary that had the right metadata and structure and wasted a whole bunch of time trying to manually create what windows already had. Windows Driver Kit (WDK) is a tool that basically does all that for you. So I installed the WDK and build the first piece of code that when passed in as a payload did not crash my system.

```c
#include <ntddk.h>

void DriverUnload(PDRIVER_OBJECT DriverObject) {
	DbgPrint("LazyCargo Analysis: Driver Unloaded Successfully!\n");
}

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
	DbgPrint("LazyCargo Analysis: Hello World! The payload executed successfully.\n");
	DriverObject->DriverUnload = DriverUnload;
	return STATUS_SUCCESS;
}
```

This was not the first attempt but this was the first payload I was able to pass in without it crashing the system. The Microsoft Driver Kit provides the build instructions so all the file formatting is done automatically along with providing the functions that are called when inside the operating system. The DriverEntry function gets called when the driver is first loaded. Passed into that are the driver object and a registry path. Then their is the unload driver function which is ment to remove the driver once it is loaded that way I could run this multiple times without having to worry about overlap besides the BYOD driver overlap.

Now due to the way that the driver is loaded I can't see the `DbgPrint()` so while the computer was no longer crashing there was no way to know if LazyCargo worked in the VM. So I continued building more complicated drivers to have definitive proof that it worked.
### Custom Payload
I chose to try and get `calc.exe` to show up on the screen after running LazyCargo as this is a very common things for pocs in exploits. Turns out that this is a very complex multi stage process to accomplish from ring 0. 



I knew the file I passed in would need to have the proper structure, since LazyCargo acts as a reflective loader for the payload. Some shell code facilitates this, so I took a closer look at what it was expecting.

The shell code was expecting a standard PE binary file. This ment that I just had to create a standard PE binary file
```c
// Code examples here
for (int i = 0; i < 10; i++) {
    printf("%d, ", i);
}
```

### Summary
Not done at the moment.
