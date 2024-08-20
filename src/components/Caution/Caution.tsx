import { FC, ReactNode } from "react";

import styles from "./Caution.module.scss";

interface Props {
  children?: ReactNode
}

export const Caution: FC<Props> = ({ children = "CAUTION" }) => (
    <div className={styles.caution}>
        <div className={styles.text}>{children}</div>
    </div>
);
