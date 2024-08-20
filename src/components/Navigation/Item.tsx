import { FC } from "react";

import { Channel } from "./Channel"
import { Direct } from "./Direct"
import { Unread } from "./Unread"

import styles from "./Item.module.scss";

interface Props {
  name: string;
  unread: number;
  active?: boolean;
  type?: "channel" | "direct"
}

export const Item: FC<Props> = ({ name, unread, active = true, type = "channel" }) => (
  <span className={`${styles.link} ${unread && styles.unread} ${active && styles.active}`}>
    {type == "direct" && <Direct name={name} />}
    {type == "channel" && <Channel name={name} />}
    <Unread unread={unread} />
  </span>
);
