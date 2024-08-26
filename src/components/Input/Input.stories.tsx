import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { Input } from "./Input";

const meta = {
  title: "Components/Input",
  component: Input,
  args: { onChange: fn() },
} satisfies Meta<typeof Input>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    type: "text",
    label: "Message",
    placeholder: "Enter your message",
    defaultValue: "Your message here",
  },
};

export const Password: Story = {
  args: {
    type: "password",
    label: "Password",
    placeholder: "Enter your password",
    defaultValue: "12345678",
  },
};

export const Number: Story = {
  args: {
    type: "number",
    label: "Number",
    placeholder: "Enter your number",
    defaultValue: 100,
  },
};
