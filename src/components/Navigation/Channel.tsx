import { FC } from "react";

import styles from "./Item.module.scss";

interface Props {
  name: string;
}

export const Channel: FC<Props> = ({ name }) => (
    <>
        <span className={styles.icon}>#</span>
        <span className={styles.element}>{name}</span>
    </>
);
