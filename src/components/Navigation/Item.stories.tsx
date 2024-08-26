import type { Meta, StoryObj } from "@storybook/react";

import { Item } from "./Item";

const meta: Meta<typeof Item> = {
  title: "Navigation/Item",
  component: Item,
  args: {},
} satisfies Meta<typeof Item>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    name: "Feeds",
    unread: 3,
    active: true,
    type: "direct",
    red: false
  },
};
