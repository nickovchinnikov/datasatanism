import { FC } from "react";

import { ChannelLink } from "./ChannelLink";

import styles from "./ChannelNav.module.scss";

interface Channel {
  id: string;
  name: string;
  href: string;
  unread: number;
  isPrivate?: boolean;
}

interface Props {
  activeChannel?: Channel;
  channels: Channel[];
}

export const ChannelNav: FC<Props> = ({
  activeChannel = null,
  channels = [],
}) => {
  return (
    <ul className={styles.nav}>
      {channels.map((channel) => (
        <li key={channel.id} className={styles.item}>
          <a
            className={`${styles.link} ${
              activeChannel && activeChannel.id === channel.id
                ? styles.linkActive
                : ""
            }`}
            href={channel.href}
          >
            <ChannelLink {...channel}>{channel.name}</ChannelLink>
          </a>
        </li>
      ))}
    </ul>
  );
};
