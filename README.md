# marcus
[*Chat with available stoics near you!*](https://marcusthestoic.streamlit.app/)   
We're talking Aurelius, not Gary.


`marcus` is a chatbot that uses OpenAI embeddings (specifically [GPT 3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5)) fine-tuned on some fundamental texts in Stoicism, including works by Seneca and Epictetus. These texts are parsed, vectorized, and stored using [Pinecone](https://www.pinecone.io) - the chatbot can query these supplementary embeddings via [Langchain tools](https://python.langchain.com/docs/modules/agents/tools/). The final product is hosted via [Streamlit](https://streamlit.io/).



Built with ❤ by [jo](https://jgdenby.github.io) and [al](https://github.com/schroeder-g)

