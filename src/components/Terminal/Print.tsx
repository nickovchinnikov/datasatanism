import React, { FC, useEffect, useState } from "react";

import { Blank } from "./Blank";
import { Cursor } from "./Cursor";
import { Message } from "./Line";

import { typeString } from "@/components/utils";

interface Props extends Partial<Message> {
  username?: string;
}

export const Print: FC<Props> = ({ username = "", label, line }) => {
  const [currentMsg, setCurrentMsg] = useState("");

  useEffect(() => {
    if (line) {
      (async () => {
        for await (const char of typeString(line)) {
          setCurrentMsg((prev) => prev + char);
        }
      })();
    }
  }, [line]);

  return line ? (
    <>
      <pre>
        {label}:~ {currentMsg}
        {currentMsg.length != line.length && <Cursor />}
      </pre>
      {currentMsg.length >= line.length && <Blank username={username} />}
    </>
  ) : (
    <Blank username={username} />
  );
};
