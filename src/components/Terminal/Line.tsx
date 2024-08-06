import { FC } from "react";

import "./Terminal.module.scss";

export interface Message {
  idx?: number;
  label: string;
  line: string;
}

export const Line: FC<Message> = ({ label, line }) => (
  <pre>
    {label}:~ {line}
  </pre>
);
