import { FC } from "react";

import styles from "./Item.module.scss";

interface Props {
  name: string;
  red?: boolean;
}

export const Direct: FC<Props> = ({ name, red = false }) => (
    <>
        <span className={`${styles.icon} ${styles.direct} ${red ? styles.red : styles.green}`} />
        <span className={styles.element}>{name}</span>
    </>
);
