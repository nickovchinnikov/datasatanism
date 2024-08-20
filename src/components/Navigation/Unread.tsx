import { FC } from "react";

import styles from "./Item.module.scss";

interface Props {
    unread: number;
}

export const Unread: FC<Props> = ({ unread }) => unread > 0 && (
    <span className={styles.element}>
        <span className={styles.badge}>{unread}</span>
    </span>
)
