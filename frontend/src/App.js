import React, { useState } from "react";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [chatHistory, setChatHistory] = useState([]); // ✅ holds full conversation

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:8000/ask", {
        question: question,
        chat_history: chatHistory, // ✅ send history to backend
      });

      console.log("FULL RESPONSE:", res.data.response);
      const newHistory = [...chatHistory, ["human", question], ["ai", res.data.response.answer]];
      const trimmedHistory = newHistory.slice(-10); // 3 pairs = 6 total entries
      setChatHistory(trimmedHistory);

      setResponse(res.data.response.answer);

      setQuestion("");  // clear input box
    } catch (err) {
      console.error(err);
      setResponse("Something went wrong.");
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>WhistleWise: Your Sports ChatBot</h1>
      <textarea
        rows={4}
        style={{ width: "100%" }}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Hey there! I'm WhistleWise! Feel free to ask any question about sports!"
      />
      <br />
      <button onClick={handleSubmit}>Submit</button>

      <h3>WhistleWise:</h3>
      <p>{response}</p>
    </div>
  );
}

export default App;
