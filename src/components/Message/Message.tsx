import { FC } from "react";

import { timesAgo } from "@/components/utils";

import styles from "./Message.module.scss";

interface Props {
  message: string;
  author: string;
  datetime: string;
}

export const Message: FC<Props> = ({ message, author, datetime }) => (
  <div className={styles.message}>
    <div className={styles.messageBody}>
      <div>{message}</div>
    </div>
    <div className={styles.messageFooter}>
      <span className="message__authoring">{author}</span> -{" "}
      {timesAgo(datetime)}
    </div>
  </div>
);
