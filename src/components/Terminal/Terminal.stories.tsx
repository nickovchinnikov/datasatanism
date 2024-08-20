import type { Meta, StoryObj } from "@storybook/react";

import { Terminal } from "./Terminal";

const meta: Meta<typeof Terminal> = {
  title: "Components/Terminal",
  component: Terminal,
} satisfies Meta<typeof Terminal>;

export default meta;
type Story = StoryObj<typeof meta>;

const prev = [
  {
    idx: 1,
    label: "nickovchinnikof_bot",
    line: "Hi! I am a bot of Nickovchinnikof. How can I help you?",
  },
  {
    idx: 2,
    label: "username",
    line: "Hello!",
  },
  {
    idx: 3,
    label: "nickovchinnikof_bot",
    line: "Hi! How can I help you?",
  },
];

const current = {
  idx: 4,
  label: "username",
  line: "Tell me what you can do?",
};

export const Primary: Story = {
  args: {
    username: "username",
    prev,
  },
};

export const Secondary: Story = {
  args: {
    username: "nickovchinnikof_bot",
    prev,
    current,
  },
};
