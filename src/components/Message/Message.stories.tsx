import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";

import { Message } from "./Message";

const meta = {
  title: "Components/Message",
  component: Message,
} satisfies Meta<typeof Message>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    message:
      "I got a gig lined up in Watson, no biggie. If you prove useful, expect more side gigs coming your way. I need a half-decent netrunner. Hit me up, provide credentials, eddies on completion.",
    author: "V. M. Vargas",
    datetime: "2024-06-01T12:00:00Z",
  },
};
