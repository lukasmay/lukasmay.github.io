+++
title = "AI Stack"
date = 2026-02-16T01:16:08-05:00
draft = false
categories = ["Deep Dive"]
tags = ["AI", "docs"]
ShowToc = true
ShowReadingTime = true
ShowBreadCrumbs = true
ShowCodeCopyButtons = true
+++

# Overview
I have been looking into self hosting LLMs and this is my attempt at putting everything that I learned on the subject down in one place (so I can stop forgetting things). Along side that I wanted to include some information about the setup that I am using to self host LLMs on my laptop and the steps that I took to build and optimize. While that is going to come in the future as their are still some things that I am changing and this is long enough already I removed some of those parts to be in the next section.

The end result of this project for me is a setup that integrates cloud AI with local LLMs to help with reverse engineering, coding, and general troubleshooting. This posts goal is to inform you about a good deal of the stuff that is going on under the hood if you want to self host LLM and build workflows around them.

# Background
When it comes to self hosting LLMs for the most part you could just download Ollama install the models and call it a day but there are lots of moving pieces to consider if you want to go beyond that. The first thing to consider when self hosting LLMs is what models you want to run so you can build the best setup for the model but to understand some of the things I am talking about in the model section that will be more towards the end.
## Hardware
There are three main things to consider when looking at hardware for running LLMs: size of model that can run, prefill speed, and inference speed. In general memory determines the size and therefore the smartness of the model you can physically run, memory bandwidth combined with compute power determine how fast the LLM that you run will be at both prefill and inference.

> NOTE: I am going to use **chip** as generic for CPU or GPU as what I am talking about applies to both.
### Memory Bandwidth
This is an important stat when looking at hardware for running LLMs due to the way that models actually run on computers. You have the weights of the model which need to be moved from memory into the chip. This means that in order to compute one step of the process a weight has to move from memory into the processor get computed and then moved back out for the next iteration. The bottle neck can either be the compute power or the bandwidth. This also starts becoming an issue when talking about clustering machines together. To increase memory bandwidth you can either: widening the "road" (bus width/channels), increasing the speed of traffic (frequency/transfer rate), and shortening the distance data travels (integration/stacking). 

Widening the road means that you add more wires that are going to transfer data between the two devices. More wires means more parallel processing due to their being more roads for traffic. But when you increase the number of channels you also need to increase the number of **unified memory controllers** (UMC) on the chip to keep up with the new channels.

The second option is to increase the speed of traffic. This essentially decreases the time between each electrical signal across each line. To accomplish this you need better signal integrity along with chipsets directly in memory to help control voltage and sync timings such as **Power Management Integrated Circuits** (PMIC) and **Clock Drivers** (CKD). These help make sure that the data is not getting degraded when sending it closer together. Another thing that can increase the "speed limit" is by changing the encoding of the data going across the wire. Two things are happening here there is the **line encoding** and **signal density**. Line encoding I am not going to get into here but essentially the important part is reducing overhead meaning sending more data with less packaging. Signal density is increasing the amount of data sent at every clock cycle across one channel by not using 1s and 0s but instead using multiple voltage levels effectively sending more data in one electrical signal.

The third thing you can do to increase memory bandwidth is to shorten the road (much more complicated then you think). The first reason todo this comes down to two things: shorter wire means less distance to travel and therefor faster trip times, and two the longer the wire the more it acts like a capacitor meaning lower frequencies for sending data due to it taking longer for electrical signals to leave the wire. To accomplish this in the hardware essentially newer chips are stacking the memory vertically so that they don't have to be flat on the board. That is called **High Bandwidth Memory** (HBM) which requires **Through-Silicon Vias** (TSVs) to connect the layers together. Modern chips are really multiple chiplets that are connected together by the motherboard. Motherboards are made of fiberglass which doesn't allow for the small connections that are needed to stack the memory in HBM. So there is a piece of silicon called the interposer that connects them allowing for the stacking of memory. That process is called **Chip-on-Wafer-on-Substrate** (CoWoS). The next thing to make the path shorter is to combine the memory of the CPU and the GPU called unified memory (mac uses this). This reduces the distance not only from the CPU or GPU and it's memory but also between the CPU and GPU. The next steps that chips are taking is building the CPU or GPU cores in with the memory scattered throughout putting the memory and processing power right next to eachother (not happening yet).

So to sum it up memory bandwidth is controlled by how many channels between the memory and chip there are, how quickly data can transfer over those channels, and how short the channels are. Memory bandwidth in most cases is the bottle neck for inference (writing speed).
### Compute Power
Another factor to consider is how powerful the chip is at computing matrix multiplication (MatMul). One of the reasons the CPU is worse for these operations compared to a GPU is that a CPU is built of a couple very complex cores that can handle lots of different things one at a time super fast while a GPU is built of lots of specialized cores that are slower but due to architecture can compute simultaneously while also being smaller. Inside the GPU there are several different types of cores for performing different tasks. When it comes to MatMul the **tensor** core is the specialist. Every time you send a prompt, the GPU must multiply your input against billions of parameters meaning billions of MatMuls. 

The reason that the tensor core is much faster at MatMuls is that it can perform the whole operation in one cycle instead of breaking it down into the multiple steps that other cores would need. This process is called **Matrix Multiply-Accumulate** (MMA) and it is able to calculate the equation D=A×B+C which is the math required when running models. There are also multiple sizes of tensor cores for different sizes of data ie precision of model weights. 

Besides the tensor cores themselves another hardware operation is happening to run LLMs faster on hardware. Sparsity as discussed in the model section needs the hardware to skip the 0 operations that the model has baked in when working with fine-grained structured sparsity. This hardware can be called sparse tensor cores. 

The CPU also plays a role in running the model even if it isn't doing the heavy lifting. The CPU is in-charge of getting the data from memory into the GPUs vRAM. If you have a fast enough CPU you can also run models that don't fully fit inside of vRAM by having some of the calculations run on the CPU. With MoE models the routing for determining what weights go into the GPU is conducted by the CPU. Additionally the CPU is what can tokenize your input and de-tokenize the output (part of the process at least). The CPU is what is also incharge of managing the KV cache. 

When looking at compute power the GPU is the workhorse that will plow through billions of MatMuls quickly but the CPU still needs to be powerful enough to feed the GPU with data and perform other management tasks when running LLMs. There are also a lot of tricks with models when it comes to getting better performance that hardware needs to be built to take advantage of.
### Memory
Really this section comes down again to speed. When using GPUs the memory that is fast is called vRAM and so how much vRAM is how large a model you can fit on your computer without seeing massive drop offs in speed. This is due to that memory bandwidth being slower when trying to communicate between CPU and GPU. This is one of the reasons that apples unified memory is so good at running LLMs. You have access to more memory that has high memory bandwidth to the GPUs. When talking about how large of a model you can run it comes down to how much memory you have that has high bandwidth to the fast compute cores. This in most cases just comes down to vRAM or the unified memory architecture of apple silicon.

> NOTE: You can split a model between vRAM and normal memory in some special cases to fit larger models. That process is called sharding.

### Landscape of Hardware
This is only important if you are looking at buying hardware to run LLMs otherwise you are just working with what you have which is the reason I am not going deep into this topic. There are really four different options when it comes to running your own LLMs with hardware. 
1. GPU builds in a PC (RTX 5090s)
2. Prebuilt desktop designed for AI (NVIDIA spark)
3. Apple silicon
4. Renting much more power full hardware in a datacenter

> Unified Memory is what Apple uses in their chips. Essentially the memory is accessible to both the CPU and GPU at the same time meaning mainly two things. You can fit much larger models then with traditional GPUs and you can do some special computational acceleration. 

## Models
When we talk about running an LLM, we are effectively talking about running a static file that contains a snapshot of intelligence. Unlike traditional software which is logic-based, a model is probabilistic. It doesn't "know" anything in the traditional sense instead it predicts the next piece of information based on the patterns it learned during training.
### Anatomy of a Model
To understand why these files are so large and require such massive bandwidth to run, we need to break down the physical composition of a Large Language Model. When you download a model whether it is a 70B parameter Llama 3 or a 671B DeepSeek you are essentially downloading a massive, serialized dictionary of matrices (tensors) and a configuration file that tells the inference engine how to stitch them together.

At the lowest level, the "file size" on your disk is dominated by the **Weights** (or parameters). When we say a model has "70 Billion Parameters," we mean it contains 70 billion individual floating-point numbers that represent the strength of connections between neurons. In a standard unquantized FP16 model, each parameter is a 16-bit floating-point number taking up 2 bytes of memory, meaning a 70B model requires roughly 140 GB of vRAM just to load. These weights are grouped into multi-dimensional arrays called tensors; for example, a single layer might have a weight tensor of size `[8192, 8192]`. To run the model, your GPU must move these massive tensors from VRAM into the compute cores to perform matrix multiplication against your input, which is why memory bandwidth is the primary bottleneck for inference speed.

If the weights are the fuel, the **Architecture** is the engine block. Almost all modern LLMs utilize the Transformer architecture, which consists of a stack of identical blocks repeated dozens of times a 70B model might have 80 of these layers. Inside each layer are **Attention Heads** and **Feed-Forward Networks (FFN)**. The Attention Heads allow the model to understand context by comparing the current token to every previous token in your prompt, while the FFN is a massive dense network where the "knowledge" is processed. In dense models, every token passes through the entire FFN, whereas Mixture of Experts (MoE) models break this FFN into smaller "experts" to save compute. These layers are interspersed with **Normalization** (like RMSNorm) to keep the mathematical values stable as they flow deep into the network.

While not part of the downloaded file, the **KV Cache** (Key-Value Cache) is a critical component of the model's anatomy during runtime. When you feed a prompt into the model, it calculates the attention values for those tokens once and stores them in VRAM so it doesn't have to recalculate them for the next word. This cache grows linearly with your context length. If you have a massive 128k context window, this "temporary" memory can easily consume more VRAM than the model weights themselves, often causing Out of Memory (OOM) errors even if the model initially loads fine.

Finally, the **Tokenizer** acts as the interface between you and the weights. Models cannot read English; they only understand numbers. The tokenizer breaks your text into chunks called "tokens" which can be words, parts of words, or spaces and assigns each one a unique ID number based on a fixed vocabulary (e.g., 128,000 unique tokens). A more efficient tokenizer can represent complex words in fewer tokens, effectively increasing your context window and generation speed.
### Model File Types
The extension on the file you download dictates how the inference engine interacts with these weights. **Safetensors** (.safetensors) is the industry-standard "raw" format, developed by HuggingFace to replace the insecure Python "Pickle" files. It is designed for speed using "memory mapping," which allows the operating system to point the model directly to the file on the hard drive without needing to copy the data into RAM first. This structure a header describing the data followed by a massive byte-stream of raw numbers makes loading massive models nearly instant on fast storage.

For self-hosting on consumer hardware, specifically via **llama.cpp**, you will use **GGUF** (.gguf). Unlike Safetensors, which often requires separate configuration files, GGUF is a binary format that packs the weights, architecture definition, quantization tables, and tokenizer into a single executable-ready file. It is specifically optimized for Apple Silicon and CPU inference, utilizing block-based quantization tables that allow the hardware to decode compressed weights on the fly with minimal overhead.

**AWQ / GPTQ:** These are specific formats optimized for running quantized models on NVIDIA GPUs. They pack the weights in a way that aligns perfectly with the GPU's memory structure for faster access. Stuff happens here.
### Quantization
If weights and architecture are the anatomy of the model, **Quantization** is the compression algorithm that makes them portable. It is arguably the single most important concept for self-hosting because it is the only reason we can run 70B parameter models on consumer hardware instead of requiring $30,000 enterprise cards.

At a high level, quantization is the process of reducing the precision of the numbers used to represent the model's parameters. Most models are trained in **FP16** (16-bit Floating Point) or **BF16** (Brain Float 16). In this format, every single weight requires 16 bits (2 bytes) of memory. This offers incredible precision, allowing for tiny nuances in the values, but it is computationally expensive and memory-hungry. Quantization takes that high-precision range and maps it to a lower-precision grid, typically **INT8** (8-bit integer) or **INT4** (4-bit integer).

Think of this like resizing a high-resolution raw image into a JPEG. You are technically throwing away data pixel-perfect color accuracy is lost but if done correctly, the human eye (or in this case, the model's reasoning capability) can't tell the difference. The "magic" of modern quantization is that neural networks are surprisingly resilient to this noise. You can often remove 75% of the information (going from 16-bit to 4-bit) while losing less than 1% of the model's intelligence.

#### The Mechanics of Precision
To understand how this works, you have to look at how the numbers are stored. In a standard 4-bit quantization, we are restricting the model to only 16 possible values (since 4 bits can only represent numbers 0-15) to approximate the infinite range of a floating-point number.

We achieve this using a **Scale Factor** and a **Zero Point**. Instead of storing the actual weight value, we store a tiny integer and a separate scaling constant that tells the GPU how to "unpack" that integer back into a jagged approximation of the original number during calculation. This is why you will often see "groups" or "blocks" mentioned in quantization settings (e.g., groupsize 128). The model doesn't just squash the entire 70B parameter set with one scale factor; it breaks the weights into small blocks (usually 32 or 128 weights) and calculates a unique scale for just that block. This localized precision allows the model to handle sensitive layers with high variance without degrading the rest of the network.
#### Smart Quantization: K-Quants and Importance
Not all parameters in a model are created equal. Some weights are "load-bearing"—they are critical for the model's logic and syntax—while others are effectively noise. If you aggressively quantize the important weights, the model becomes lobotomized (perplexity increases). If you leave the useless weights at high precision, you are wasting vRAM.

This is where modern formats like **GGUF** and its **K-Quants** (K-series) come into play. When you see a file labeled `Q4_K_M`, it isn't just a flat 4-bit truncation. It uses a smart, mixed-precision approach called **superblocking**. The "K" refers to the specific quantization algorithm (often involving k-means clustering) that optimizes how these quantization levels are assigned. In a `Q4_K_M` model, the attention mechanisms (the most sensitive part of the brain) might be kept at 6-bit precision, while the feed-forward layers (the bulk storage) are dropped to 3-bit or 4-bit. This allows you to fit a massive model into a smaller vRAM footprint while keeping the "smart" parts sharp.

#### Activation-Aware Quantization (AWQ)
While GGUF is king for Apple Silicon and CPU inference, **AWQ** (Activation-aware Weight Quantization) has become the standard for high-performance GPU serving. The breakthrough of AWQ was the realization that you shouldn't just look at the weights to decide what to compress you should look at the activations.

During the quantization process, AWQ feeds a small amount of calibration data into the model to see which weights actually "light up" or activate the most during inference. It identifies the 1% of salient weights that are crucial for performance and protects them, keeping them in higher precision or scaling them differently, while aggressively compressing the other 99%. This results in models that are significantly faster and more accurate than older methods (like GPTQ) because they preserve the specific pathways the model uses to think, rather than just preserving the static map of the weights.

#### KV Cache Quantization
Finally, we have the new frontier: **KV Cache Quantization**. Traditionally, even if you compressed your model weights to 4-bit, the runtime memory (the context window) was still stored in massive FP16. For a long conversation, this temporary memory could easily grow larger than the model itself. New techniques now allow us to quantize this temporary cache into FP8 or even INT4. This creates a slight degradation in "recall" (the model might forget a specific detail from 100 pages ago), but it allows for massive context windows effectively letting you fit a 128k token context into the same space that used to hold only 8k.

### Tool Aware Models
If standard LLMs are the "brains" that think and reason, **Tool Aware** (or Function Calling) models are the brains connected to hands. A standard base model is effectively trapped in a text-only box it can tell you how to check the weather, but it cannot actually check it. Tool aware models bridge this gap, transforming the LLM from a passive chatbot into an active agent that can interact with your operating system, APIs, and local files.
#### The Mechanism: Function Calling
At a technical level, "using a tool" is really just a structured game of fill-in-the-blanks. When you load a tool-aware model, you don't just send it a user prompt you also send it a list of available functions (tools) defined in a schema (usually JSON via MCP).

For example, you might provide a tool definition for `get_current_weather(location: string)`. If you ask the model "What's the weather in Paris?", a standard model would hallucinate an answer or say "I don't know." A tool-aware model, however, detects that the user's intent matches one of its available tools. Instead of generating conversational text, it flips a switch and generates a **Structured Output** typically a JSON block looking like `{"function": "get_current_weather", "parameters": {"location": "Paris"}}`.
#### The Agentic Loop
It is important to understand that the model itself never runs the code. It simply writes the request. The "magic" happens in the orchestration layer (the software running the model, like Ollama, vLLM, or LangChain).
1. **Reasoning:** The model analyzes the prompt and decides it needs external data. It outputs the specific "Tool Call" token and the JSON command.
2. **Execution (The Pause):** The inference engine detects this stop token, pauses the model generation, and takes that JSON payload. It executes the actual Python script or API call on your machine.
3. **Observation:** The engine takes the return value of that function (e.g., "Temp: 15°C, Rainy") and feeds it back into the model's context window as a "Tool Result."
4. **Response:** The model "wakes up," sees the result of the tool it requested, and uses that new fact to generate the final answer for the user.

> **Key Distinction:** While you can prompt almost any model to output JSON, true **Tool Aware** models (like Llama 3.1, Hermes, or Command R) are fine-tuned specifically on massive datasets of function interactions. They are significantly less likely to hallucinate non-existent parameters or mess up the JSON syntax, which is critical for building reliable automated workflows.

### Architectures: Dense vs. Mixture of Experts (MoE)
The architecture of a model determines how efficiently it turns raw parameters into intelligence. **Dense Models**, like Llama 3 or Gemma, represent the traditional approach to AI structure. In these models, every single parameter is active for every single token generated. It is effectively a brute-force method where the entire neural network is engaged to answer even the simplest query. While this ensures consistency, it comes with a steep cost in compute and memory bandwidth, as your hardware must move the entire weight set through the GPU cores for every word produced.

**Mixture of Experts (MoE)** models, such as Mixtral, Qwen, or DeepSeek, introduce a concept called sparsity to solve this efficiency problem. Instead of one giant monolithic neural network, the model is fragmented into smaller sub-networks known as "experts." A router layer sits at the front of the process, analyzing each token and activating only the specific experts needed for that concept perhaps one expert for syntax and another for factual recall. This architecture allows for a massive disconnect between the file size and the run cost. An MoE might have 47 billion total parameters on disk, but only use 13 billion active parameters during inference. This gives you the broad knowledge base of a massive model with the speed and responsiveness of a much smaller one.
### The Impact of Size and Scaling Laws
When we talk about model size, we are really talking about the capacity for complexity. Parameters function as storage slots for patterns, and the number of parameters dictates how deep those patterns can go. Smaller models, typically under 10 billion parameters, have enough capacity to master English grammar, basic facts, and surface-level instruction following. They are excellent for summarization but often lack the depth to handle multi-step logic without losing the thread of the conversation.

As you scale up to the 30B and 70B parameter range, you start to see emergent behaviors in reasoning. This is the threshold where models move beyond simple pattern matching and start to understand nuance, solve logic puzzles, and handle complex coding tasks with significantly fewer hallucinations. They can maintain a coherent train of thought over much longer conversations. Once you push past the 100B parameter mark, the model gains deep world knowledge and the ability to generalize across domains it wasn't explicitly trained on, though this comes with the hardware cost mentioned earlier.

### Model Specializations
It is also critical to understand that not all models are trained for the same purpose. The raw output of a training run is called a **Base Model**. These are not chatbots; they are text completion engines designed to predict the next word in a sequence. If you ask a base model a question, it is just as likely to generate five more questions as it is to answer you, because it views the input as a pattern to be continued rather than a query to be resolved. These are generally useless for standard chat applications but are the preferred foundation for researchers fine-tuning their own datasets.

For almost all practical applications, you will want an **Instruct** or **Chat** model. These are base models that have undergone Reinforcement Learning from Human Feedback (RLHF) to understand the "User asks, Assistant answers" dynamic. Beyond standard chat, we are now seeing the rise of **Reasoning Models** (like DeepSeek R1), which are trained with Chain of Thought data to "talk to themselves" and error-check their logic before responding, and **Coding Models**, which are fine-tuned on massive repositories of code to understand syntax and edge cases that general models often miss.

### Summary
Models are very complex. There are dense models and mixture of expert models. Quantiziation is just shrinking of models to fit on worse hardware without loosing to much of the models ability. The model is really just a bunch of weights and some instructions on what they mean or how to work with them. There is not one format for models to be stored, quantized, built, or ran.

## Inference Engine
So far the hardware that runs the models and the models themselves have been explained (mostly). The inference engine is what ties these to together. It is responsible for running the model at the software level. Inference engines are responsible for loading the model, managing memory, scheduling requests, and executing MatMul. All the speed tricks in the models and on the hardware need the inference engine be able to use them in-order to gain the massive benefits that they provide. 

### Workflow Execution
To run a model the inference engine needs to follow several steps to go from input prompt to output.
1. Load the model from disk into memory if not already there
2. Tokenization: Convert your input text into tokens ie turn text into numbers
3. Prefill: Builds initial KV cache from your prompt. Computationaly heavy but only happens once
4. Decode: This is the generation part. The engine loops generating one token at a time.
5. Detokenization: Takes the tokens generated and turns them back into words
### Memory Management
One of the optimizations when it comes to memory management is Paged Attention. The problem that is solves is the allocation of the KV cache to be more dynamic and granular instead of setting aside one large contiguous space in vRAM it uses blocks. This means you have dynamic allocation and on demand growth for context size. This leads to much less waisted space if you have multiple queries running simultaneously. 

With this you also gain more throughput by having more granular control. If one part of the process finishes first the memory can be freed up for the next step in the process instead of needing to wait for the whole section to complete before moving on. This really only applies to multiple requests simultaneously from the user although with more agentic workflows that is the reality of what is happening under the hood. That process is called Continuous Batching although it requires page attention to work.

Another benefit of Page Attention is memory sharing. If you ask the LLM to come up with multiple solutions the inference engine can store the prompt tokens in one place and use them for all three iterations with the model saving massively on space. Also if you want your LLM to try different options the different branches can use the same technique to store their shared context via a process called Beam Search.  If there are any changes that need to happen to the shared KV cache you just copy the block that is different keeping the rest shared leading to massive space saving results.
### Parallelism
This is the process of splitting up the model across multiple GPUs. When done correctly this gives you more compute and memory bandwidth making the process faster while also giving you more space to fit a model in the fast vRAM of the combined GPUs. For the most part there are two types of parallelism: Tensor parallelism, and pipeline parallelism. 

Tensor Parallelism essentially splits a models layer across multiple GPUs. This method allows all the compute power of the GPUs to work on the same step of the inference process at the same time. In this process all the GPU cores need to sync backup in between each layer. The math that is actually split is the matrix itself but the summing part of the equation is what needs to be synced across all the cores. This means that high bandwidth is critical as there is a lot of communication in between each chip. Pros of this setup is single user generation speed. Cons are you need really fast connection speeds to make sure that the GPUs are all getting saturated.

Pipeline parallelism splits the model vertically between GPUs. Meaning different layers are on different GPUs. The reason todo this is if you need a larger model that won't fit on one GPU and the memory bandwidth is not great. The reason for this is that only one GPU will actually be working at a time but it does decrease the amount of communication the GPUs need todo to function together.

> In data centers both tensor and pipeline parallelism are used in conjunction to link GPUs in a rack together and then different server racks together respectively.

### Hardware Translator
The models just contain the weights of the model but the hardware needs the instructions on what todo with those numbers. The inference engine pieces of code that know how to interact with the hardware drivers to tell the GPU how to compute the results. This means that the inference engine has to know the best way to utilize the hardware that it is running on in-order to get the best results as hardware differs. 

The optimizations that can be made here are mainly what instructions the engine sends to the GPU and the best choice for that depends on all the aspects of the hardware that is being used. So getting the inference engine to run optimally requires a lot of testing and research about what the best hardware instructions work for the hardware currently in use. The brand of hardware also changes what functions are available.

### Parameters
While each inference engine has different parameters there are some general controls that are good to know exist. There are two general categories of parameters the load time parameters which control how the model is loaded into the hardware and handles the memory and runtime/sampling parameters which control behavior of the output.

> Flags for these parameters are different depending on the engine so I am not going to list the specific flag names

For load time parameters one of the main ones to work with is the flags for controlling the offloading or splitting of the model in the hardware. These flags are what you use to control how the parallelism is setup. Another parameter controls the context window and memory reservation. The KV cache takes up a lot of space and grows linearly with context growth so you might have to limit the max context size or increase it from defaults if the hardware and model can handle it. You can also control the max amount of memory that can be allocated depending on how dangerous you want to play it. There is also the option to optimize the KV cache by effectively quantizing the cache to smaller precision.

Runtime parameters are less about controlling performance and more about improving the output quality. An important parameter here is called temperature which effectively controls the randomness of responses. The higher the temperature the more "creative" the model becomes. Nucleus Sampling is another flag that enables a process the lops off the lesser probable options and another version of it just ignores tokens that are below a specified probability. Another section of runtime parameters control the response structure. One of them is a limit on the number of tokens one request will generate. You can also create penalties to stop looping situations with smaller models. Depending on the model this might not be a good choice as it might need to repeat itself internally for longer thinking but it can be helpful on occasion.

Those are some of the general areas of control that you have access to although there are a lot more and depending on the engine their might be unique parameters. The bottom line is that the parameters are where almost all the optimization comes in once you have settled on hardware.
### Landscape of Inference Engines
There are a lot of different inference engines that are free to use with the main ones being: llama.cpp, vLLM, MLX, and TensorRT-LLM. Each of them has different uses and optimizations.

llama.cpp is the general worker that is built to run on all types of systems and hardware. It is the backend inference engine for ollama the common self hosting tool which handles downloading the models and other useful features. llama.cpp is also designed to run on consumer hardware ie the cheaper end of the spectrum where their are fewer of the hardware tricks in place. An important thing to note is that this inference engine runs GGUF models (the creator of llama.cpp also created GGUF). Due to it being made to run on more consumer hardware it has a lot of features and compatibility with sharding the model across the GPU and CPU. It has compatability with apples metal hardware API, nvidia cuda drivers, AMD drivers, and AVX-512 instructions. With this in mind llama.cpp is one of the best choices regardless of what hardware that is available.

vLLM is a inference engine for better hardware. It runs AWQ / GPTQ / FP8 models and came up with paged attention. It also has great support for tensor parallelism as it is designed for much more compute then llama.cpp is focused on. Due to it being more for enterprise grade equipment it also has tool calling built into the engine with some very complex workflows that allow vLLM to run multiple tool calls simultaniously streamlining some processes. A feature of vLLM is that through some witchcraft it can store the KV cache in RAM instead of in vRAM allowing massive context windows to run with limited speed degration. One thing that vLLM doesn't do is work well with offloading weights to the CPU due to that not being a focus for the tool.

MLX is the inference engine developed my Apples machine learning team specifically to run LLMs on Apples hardware. This means for the most part that if you have Apple silicon this is the best option as it takes advantage of some hardware specifics. The main thing is the unified memory architecture that I talked about in the hardware section briefly. Due to both CPU and GPU using the same memory their is no need to copy any data from CPU to GPU instead just hand off the location of the data from the CPU to the GPU. This means that their is significantly less copying of data. It also uses something called Lazy evaluation which is essentially a process where instead of immediately calculating results it waits until you need to output to calculate it. This gives the engine more time to combine the instructions that it sends to the hardware to complete the operations leading to drastically reducing overhead on the GPU and keeps it more saturated allowing for faster results. Then there is the fact that MLX utilizes all the hardware to get the best results so the GPU, CPU, and ANE get used. Another thing that this inference engine has is the ability to train or fine tune models to your use cases which I am not going to cover here as that is a whole other can of worms I am still exploring. 

TensorRT-LLM is the MLX for NVIDIA. It designed to be very fast. If you have NVIDIA GPUs that can run it. Do. The quick overview is that it takes in the model and then optomizes it for whatever hardware you have through Automatic Inference Optimization and Layer Fusion. Through the Graph Rewriting and Specialized Precision Handling it is able to change the model to run the best on your setup in a pre-compilation phase. It also has great support for tensor parallelism and communication between GPUs. This is the inference engine used in NVIDIA based data centers for running cloud AI. It only runs on NVIDIA hardware.

> To summarize the inference engine is responsible for the running and operations of the model. This is where the optimizations happen in self hosting because the right engine can use the hardware to the greatest effect.

## MCP & RAG
Both of MCP and RAG are tools that you can add to your setup to allow more capabilities and ensure better results. MCP is a standard protocol that allows your LLM todo things in a system. RAG is a system to give your tool more context about the situation related to the prompt. Putting them together can lead to the best results when you want an LLM to understand the specific problem, be able to use tools outside itself, and make changes.

### MCP Components
The MCP client is the component that is built into the chat interface used to interact with the LLM. The clients job is to inform the LLM about the tools avaliable, actually call the tool the LLM wants to call, and if any information comes back from that call inform the LLM about the output. It does this by just inserting text into the prompt after you hit enter but before it reaches the LLM. This text informs the LLM how to call tools which normally are just the LLM outputing the text in a tool call format ie `tool_get_data()`. Depending on what tool is called the client then forwards the call off the the MCP server.  If the server has a response it again injects that text data into the context window of the LLM essentially giving it the knew knowledge or information.

MCP servers job is to wrap a tool in a way that is easy for the LLM to interact with. The server is an API for the LLM to use a tool. When building them it is important to remember to build them in a way that is best for the LLM (not in standard API format). When the MCP client sends the request off to the server the servers job is to run the command or function that is requested and then to send the results back to the client to be injected into the LLM. This is the component that you configure when setting up MCP.

> The integration of the client MCP server and the inference engine change a little if the model is tool aware as it might put the data in a separate context window but that is handled by the inference engine.

### MCP Best Practices
Some things to consider when working with MCP servers. First is that for most tools there is already a MCP server online that you can use. I recommend running that through MCP-scan or proximity to make sure you are not installing malware. Additionally a lot of them are not very good and depending on what you want from the tool sometimes it is easier to build your own for the specific task you want the LLM todo. If that is the case keep in mind these things. Expose outcomes not operations. Make it as simple as possible for the LLM to get the job done or the data it needs. This means creating higher level tools that do more then a standard API function. Limit the tools that you expose as the more tools you have the more context windows it takes up. Ensure parameters are flat data types ie strings and simple so that the LLM has less of a chance to make a mistake when calling the tool. Constrain variables to enums so that the LLM has less chance to messing up the tool call. Use the doc strings of the functions to explain what the tool does in a way that an LLM can understand. Make sure that errors return more context then just an error code to help the LLM understand more about what went wrong. Lastly one of the most important curate the data that is getting returned. Return only the data that is relevant instead of dumping massive amounts of json. Also security wise make sure that you have limitations on what the AI can do so that it doesn't accidentally delete your the system.

### MCP Primitives
Tools are the most well known primitive inside of MCP servers. The server defines a tool with a name, description, and input schema (parameters). The LLM analyzes the description and, if it decides the tool is helpful, generates a call with the necessary arguments. The model chooses when to call them.

Resources are just documents that provide context to LLMs. These are defined information that you might want to bring in when working with MCP. This can act similarly to skills in Claude code. The purpose is to provide the model with background information, logs, or file contents needed to answer a question. The front end chooses when to call them.

Prompts are templates or workflows that you can define within your MCP server that the user calls. The goal is to help users use the server effectively by providing pre-built instructions or standard operating procedures. If you find that prompting the LLM in a specific way gets the best results with a specific tool you could add a prompt resource to speed up that process. 

Tasks are in some MCP servers and they are essentially a structure that goes beyond a prompt that is ment for long thinking and complex multi-step workflows. Tasks allow for tracking progress, pausing/resuming execution, and managing durable workflows
### What is RAG?
**Retrieval-Augmented Generation** (RAG) is a technique used to enhance the accuracy and reliability of LLMs by fetching relevant data from an external knowledge base before generating a response. This helps solve the problem of the AI not knowing about the specific topic you are talking about by augmenting the LLMs context window with relevant information.

The typical RAG workflow involves passive retrieval followed by generation:
1. Query: The user asks a question
2. Retrieval: The system searches a specific knowledge base for relevant text chunks.
3. Augmentation: The retrieved text is combined with the original question into a single prompt
4. Generation: The LLM generates an answer based on the provided facts rather than its training data.
### How does RAG work?
Now the magic of RAG is in the retrieval process how it knows what is relevant to your prompt. The fist step is converting the data being drawn from and the prompt into something called **embeddings**. The embeddings are really just dense semantic vectors that make up the sentiment of each data point. This allows semantic matching instead of just text matching giving far better results. 

There is the vectorized database of information you want to have available to the LLM. Whenever a prompt is sent to the LLM the RAG system will semanticly vectorize the input and look for close matches in the database. Depending on the parameters of the setup the closer results are then uploaded into the models context back as the data. 
## Automation
There are several automation tools to help with building more agentic workflows. n8n is one of the leading general purpose tools that allows you to build complex workflows with multiple LLMs working together. n8n allows you to build a decision matrix for a path that allows the LLM to chose what todo within the outlined options. This tool also allows different models to interact allowing for the specialties of each model to be applied to the greatest effect. n8n can also be run using docker so it is really easy to spin up and setup a test workflow. n8n is the general workhorse so great for testing but not for optimization.

LangGraph is another tool that falls into thinking tools. It creates complex graph-based stateful multi agent workflow utilizing each model to its fullest. This tool is ideal for processes that require lots of thinking. 

Gumloop falls under the category of automation tools built for heavy data driven processes. It moves lots of data across SaaS apps and suports MCP and natural language building of workflows. Not a total fan but it does a good job of taking in lots of data.

There are many other orchestration or automation tools out there that help build better more advanced agentic workflows just give it a google I have not spent to much time picking these for my setup or researching them.

## Other Tooling
- LM Studio: Front end that can use either llama.cpp or MLX
- exo: clustering orcestrator to speed up results (Would use if clustering)
- openwebui: chatgpt like interface for interacting with inference engine (what I use)
- opencode: terminal based front end that has lots of cool features (what I use)

# Summary
There are a lot of moving parts that when done right can all link up and provide massive performance benefits. The largest improvements in speed or ability to run larger models comes from the hardware but the software on top if it need to utilize the specific tricks in each setup to get those results. The information that I covered here is important for the next steps which is picking the best options now with the knowledge of what to look for when trying to run local LLMs.
