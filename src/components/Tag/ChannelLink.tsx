import { FC } from "react";

import { Badge } from "./Badge";

import styles from "./ChannelLink.module.scss";

interface Props {
  name: string;
  unread: number;
}

export const ChannelLink: FC<Props> = ({ name, unread }) => (
  <span className={`${styles.link} ${unread && styles.unread}`}>
    <span className={styles.icon}>#</span>
    <span className={styles.element}>{name}</span>
    {unread > 0 && (
      <span className={styles.element}>
        <Badge>{unread}</Badge>
      </span>
    )}
  </span>
);
