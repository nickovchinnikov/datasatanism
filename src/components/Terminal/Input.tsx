import React, { useState, useEffect } from "react";

import { Cursor } from "./Cursor";

interface Props {
  username: string;
  onEnter: (input: string) => void;
}

export const Input: React.FC<Props> = ({ username, onEnter }) => {
  const [inputBeforeCursor, setInputBeforeCursor] = useState("");
  const [inputAfterCursor, setInputAfterCursor] = useState("");

  const handleKeyDown = (event: KeyboardEvent) => {
    switch (event.key) {
      case "Enter":
        onEnter(inputBeforeCursor + inputAfterCursor);
        setInputBeforeCursor("");
        setInputAfterCursor("");
        break;
      case "ArrowLeft":
        if (inputBeforeCursor.length > 0) {
          setInputAfterCursor(inputBeforeCursor.slice(-1) + inputAfterCursor);
          setInputBeforeCursor(inputBeforeCursor.slice(0, -1));
        }
        break;
      case "ArrowRight":
        if (inputAfterCursor.length > 0) {
          setInputBeforeCursor(inputBeforeCursor + inputAfterCursor.charAt(0));
          setInputAfterCursor(inputAfterCursor.slice(1));
        }
        break;
      case "Backspace":
        setInputBeforeCursor(inputBeforeCursor.slice(0, -1));
        break;
      case "Delete":
        setInputAfterCursor(inputAfterCursor.slice(1));
        break;
      default:
        if (event.ctrlKey) {
          console.log("ctrl key pressed");
          switch (event.key) {
            case "c":
              // Implement copy logic here
              // You might need to use the Clipboard API or store the selected text in state
              break;
            case "v":
              // Implement paste logic here
              // Use the Clipboard API to retrieve text from the clipboard
              console.log("ctrl+v");
              navigator.clipboard.readText().then((clipText) => {
                console.log("Clipboard text:", clipText);
                setInputBeforeCursor(inputBeforeCursor + clipText);
              });
              break;
            case "a":
              // Implement select all logic here
              // This might involve setting some state to indicate all text is selected
              break;
            default:
              break;
          }
        }
        if (event.key.length === 1) {
          setInputBeforeCursor(inputBeforeCursor + event.key);
        }
        break;
    }
  };

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  });

  return (
    <pre>
      {username}:~ {inputBeforeCursor}
      <Cursor />
      {inputAfterCursor}
    </pre>
  );
};
