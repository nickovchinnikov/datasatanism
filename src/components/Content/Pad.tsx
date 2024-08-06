import { FC, ReactNode } from "react";

import styles from "./Content.module.scss";

interface Props {
  children: ReactNode;
}

export const Pad: FC<Props> = ({ children }) => (
  <div className={styles.pad}>
    <div className={styles.padBody}>{children}</div>
  </div>
);
