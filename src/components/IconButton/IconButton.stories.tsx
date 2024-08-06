import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { IconButton } from "./IconButton";

const meta = {
  title: "Components/IconButton",
  component: IconButton,
  argTypes: {
    size: {
      control: {
        type: "number",
        min: 2,
        max: 10,
        step: 1,
      },
    },
  },
} satisfies Meta<typeof IconButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    icon: "Moon",
    onClick: fn(),
    size: 2,
  },
};
