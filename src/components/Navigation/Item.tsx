import { FC, ReactNode } from "react";

import { GlitchButton } from "@/components/GlitchButton/GlitchButton"

import { Channel } from "./Channel"
import { Direct } from "./Direct"
import { Unread } from "./Unread"

import styles from "./Item.module.scss";

export interface Props {
  name: string;
  children: ReactNode;
  unread?: number;
  active?: boolean;
  type?: "channel" | "direct"
  red?: boolean;
}

export const Item: FC<Props> = ({ name, unread, active = true, type = "channel", red = true }) => (
  <span className={`${styles.link} ${unread && styles.unread} ${active && styles.active}`}>
    {type == "direct" && <Direct red={red} name={name} />}
    {type == "channel" && <Channel name={name} />}
    <Unread unread={unread} />
  </span>
);
