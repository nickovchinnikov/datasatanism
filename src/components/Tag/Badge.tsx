import { FC, ReactNode } from "react";

import styles from "./Badge.module.scss";

interface Props {
  children: ReactNode;
}

export const Badge: FC<Props> = ({ children }) => (
  <span className={styles.badge}>{children}</span>
);
