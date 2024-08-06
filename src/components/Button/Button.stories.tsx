import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { Icon } from "@/components/Icons";
import { Button, Props } from "./Button";

const meta: Meta<Props> = {
  title: "Components/Button",
  component: Button,
  args: { onClick: fn() },
} satisfies Meta<typeof Button>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    children: "Button",
    variant: "primary",
    type: "button",
    size: "lg",
    onClick: fn(),
  },
};

export const IconButton: Story = {
  args: {
    children: <Icon name="Moon" size={2} />,
    variant: "icon",
    type: "reset",
    size: "lg",
    onClick: fn(),
  },
};
