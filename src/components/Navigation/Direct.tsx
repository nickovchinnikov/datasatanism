import { FC } from "react";

import styles from "./Item.module.scss";

interface Props {
  name: string;
}

export const Direct: FC<Props> = ({ name }) => (
    <>
        <span className={`${styles.icon} ${styles.direct}`} />
        <span className={styles.element}>{name}</span>
    </>
);
