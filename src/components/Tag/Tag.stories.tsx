import React from "react";

import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { Tag } from "./Tag";
import { ChannelLink } from "./ChannelLink";
import { ChannelNav } from "./ChannelNav";

const meta = {
  title: "Components/Tag",
  component: Tag,
  args: {},
} satisfies Meta<typeof Tag>;

export default meta;
type Story = StoryObj<typeof meta>;

const feed = [
  { id: "5ba5", name: "Afterlife", href: "#", unread: 3 },
  { id: "4f22", name: "NCPD-Gigs", href: "#", unread: 0 },
  { id: "fee9", name: "Pacifica", href: "#", unread: 0 },
  { id: "a0cc", name: "Watson", href: "#", unread: 0 },
  { id: "dee3", name: "_T_SQUAD", href: "#", isPrivate: true, unread: 2 },
];

const activeChannel = feed[0];

export const Primary: Story = {
  args: {
    renderTitle: (props) => <h2 {...props}>Direct</h2>,
    children: <ChannelNav activeChannel={activeChannel} channels={feed} />,
  },
};
