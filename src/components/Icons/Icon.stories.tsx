import type { Meta, StoryObj } from "@storybook/react";

import { Icon } from "./index";

const meta = {
  title: "Components/Icon",
  component: Icon,
} as Meta<typeof Icon>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    name: "Home",
  },
};
