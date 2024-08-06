import { FC } from "react";

import { Cursor } from "./Cursor";
import styles from "./Terminal.module.scss";

export const Blank: FC<{ username: string }> = ({ username }) => (
  <>
    <div className={styles.scanline}></div>
    <pre>
      {username}:~ <Cursor />
    </pre>
  </>
);
