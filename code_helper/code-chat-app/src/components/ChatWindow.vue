<template>
    <div class="flex flex-col h-screen p-4 bg-gray-100">
      <div class="flex-1 overflow-y-auto p-2 bg-white rounded shadow-md mb-4">
        <div v-for="(message, index) in messages" :key="index" class="mb-2">
          <div :class="message.role === 'user' ? 'text-right' : 'text-left'">
            <div v-if="message.role === 'assistant'" class="bg-slate-100 inline-block p-2 rounded">
              <pre v-html="formatMessage(message.content)" class="whitespace-pre-wrap code-block"></pre>
            </div>
            <div v-else class="bg-blue-500 text-white inline-block p-2 rounded">
              <p class="whitespace-pre-wrap">{{ message.content }}</p>
            </div>
          </div>
        </div>
      </div>
      <div class="flex">
        <textarea
          v-model="newMessage"
          @keydown.enter="handleEnter"
          @keyup.enter="handleEnter"
          class="flex-1 p-2 rounded-l border-t border-b border-l border-gray-300"
          placeholder="Type a message..."
        ></textarea>
        <button @click="sendMessage" class="p-2 bg-blue-500 text-white rounded-r">Send</button>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios';
  import hljs from 'highlight.js';
  import 'highlight.js/styles/atom-one-dark.css';  // Import the dark theme CSS

  export default {
    data() {
      return {
        newMessage: '',
        messages: [],
      };
    },
    methods: {
      async sendMessage() {
        if (this.newMessage.trim() === '') return;
  
        const userMessage = {
          role: 'user',
          content: this.newMessage,
        };
  
        this.messages.push(userMessage);
        this.newMessage = '';
  
        try {
          const response = await axios.post('http://192.168.0.4:5000/generate', {
            content: userMessage.content,
          });
  
          const assistantMessage = {
            role: 'assistant',
            content: response.data.response,
          };
  
          this.messages.push(assistantMessage);
        } catch (error) {
          console.error('Error sending message:', error);
        }
      },
      handleEnter(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          this.sendMessage();
        }
      },
      formatMessage(content) {
        const codeBlockPattern = /```(\w+)?\n([\s\S]*?)```/g;
        let formattedContent = content.replace(codeBlockPattern, (match, lang, code) => {
          if (lang && hljs.getLanguage(lang)) {
            const highlightedCode = hljs.highlight(code, { language: lang }).value;
            return `<pre><code class="hljs ${lang}">${highlightedCode}</code></pre>`;
          } else {
            const highlightedCode = hljs.highlightAuto(code).value;
            return `<pre><code class="hljs">${highlightedCode}</code></pre>`;
          }
        });
        return formattedContent;
      },
    },
  };
</script>
  
<style>
  body {
    @apply bg-gray-100;
  }
  
.hljs {
    padding: 1em;
    border-radius: 0.5em;
    overflow-x: auto;
    background-color: #282c34; /* Darker background for code blocks */
  }
  
  
.chat-container {
    @apply bg-gray-200;
  }
</style>
  