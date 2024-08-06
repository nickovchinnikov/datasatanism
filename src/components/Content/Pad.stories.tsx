import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { Pad } from "./Pad";

const meta = {
  title: "Components/Pad",
  component: Pad,
  args: {},
} satisfies Meta<typeof Pad>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    children: `A fake Slack or Discord type of app inspired by Cyberpunk 2077. This app is static, eg. not implementing much logic.

    The goal is: showcasing a start of a UI kit. If you've played the game, you' might be able to pick-up some similarities with the in-game menus.`,
  },
};
