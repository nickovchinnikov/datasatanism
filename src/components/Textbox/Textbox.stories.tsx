import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { Textbox } from "./Textbox";

const meta = {
  title: "Components/Textbox",
  component: Textbox,
  args: { onChange: fn() },
} satisfies Meta<typeof Textbox>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    label: "Message",
    defaultValue: "",
  },
};

export const DefaultValue: Story = {
  args: {
    label: "Message",
    defaultValue: `The goal is: showcasing a start of a UI kit. If you've played the
    game, you' might be able to pick-up some similarities with the
    in-game menus.
    I got a gig lined up in Watson, no biggie. If you prove useful, expect more side gigs coming your way. I need a half-decent netrunner. Hit me up, provide credentials, eddies on completion.`,
  },
};
