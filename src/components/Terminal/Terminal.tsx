import React, { FC } from "react";

import { Line, Message } from "./Line";
import { Print } from "./Print";
import { Input } from "./Input";

interface Props {
  prev: Message[];
  current?: Message;
  username?: string;
}

export const Terminal: FC<Props> = ({
  prev = [],
  current = null,
  username = "",
}) => (
  <>
    {prev.map(({ idx, label, line }) => (
      <Line key={idx} label={label} line={line} />
    ))}
    {current ? (
      <Print username={username} label={current?.label} line={current?.line} />
    ) : (
      <Input username={username} onEnter={() => {}} />
    )}
  </>
);
