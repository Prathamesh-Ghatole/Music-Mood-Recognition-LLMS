# Enhanced Mood and Theme Recognition in Music Using Lyrical Sentiment Analysis with LLMs and Audio Signal

## Abstract:
Conventional approaches to music mood analysis often neglect the profound semantic layers embedded within song lyrics, primarily focusing on audio signal. This work introduces an innovative method to advance mood and theme recognition in music by not only integrating lyrical sentiment analysis with digital signal processing, but also taking it to the next level using cutting edge LLMs. Deep Learning algorithms are then employed to combine the lyrical and audio signal understanding to extract accurate mood and theme related information from songs.

## Objectives:
- Develop an integrated approach to music mood analysis by combining lyrical sentiment analysis, digital signal processing, and Large Language Models (LLMs).
- Enhance understanding of song lyrics' emotional and cultural landscape through LLMs, enabling nuanced analysis of informal and unconventional language.
- Identify recurring themes and topics within song lyrics to unveil deeper insights into underlying narrative and artistic intent, improving mood classification precision.

## Methodology:
Our implementation focuses on a hybrid approach to mood and theme recognition by integrating LLMs with existing audio signal analysis, and combining the results using cutting edge deep learning technologies like CNNs and LSTMs.

The over-arching structure of our implementation can be observed in the following diagram:
<br>

![diagram](https://i.imgur.com/ivx5K5V.png)

<br>

## Setup:
Download and store the following files:
- [PMEmo2019 Dataset](https://drive.google.com/drive/folders/1qDk6hZDGVlVXgckjLq9LvXLZ9EgK9gw0) -> Extract zip to `./data/PMEmo2019/`
- [llama-2-7b-chat.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf?download=true) -> Extract to `./models/llama-2-7b-chat.Q4_0.gguf/`
- [mistral-2-7b-chat.Q4_0.gguf](https://huggingface.co/TheBloke/Mistral-2-7B-Chat-GGUF/resolve/main/mistral-2-7b-chat.Q4_0.gguf?download=true) -> Extract to `./models/mistral-2-7b-chat.Q4_0.gguf/`
- [Phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf?download=true) -> Extract to `./models/Phi-3-mini-4k-instruct-q4.gguf/`

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage:
Run the following command to start the application:
```bash
streamlit run app.py
```