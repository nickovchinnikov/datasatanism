import React, { FC, ReactNode } from "react";

import { Badge } from "./Badge";
import { ChannelLink } from "./ChannelLink";

import styles from "./Tag.module.scss";

interface Props {
  children: ReactNode;
  renderTitle: FC<{ className?: string }>;
}

export const Tag: FC<Props> = ({ children, renderTitle }) => {
  return (
    <div className={styles.section}>
      <div className={styles.header}>
        {renderTitle({ className: styles.title })}
      </div>
      <div className={styles.body}>{children}</div>
    </div>
  );
};
