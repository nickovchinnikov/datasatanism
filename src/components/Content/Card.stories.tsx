import React from "react";
import { StoryObj, Meta } from "@storybook/react";

import { Card } from "./Card";

const meta = {
  title: "Components/Card",
  component: Card,
  args: {
    head: "What's this?",
    blocks: [
      {
        key: "1",
        value: (
          <>
            A <em>fake</em> Slack or Discord type of app inspired by Cyberpunk
            2077. This app is static, eg. not implementing much logic.
          </>
        ),
      },
      {
        key: "2",
        value: (
          <>
            The goal is: showcasing a start of a UI kit. If you&apos;ve played
            the game, you&apos; might be able to pick-up some similarities with
            the in-game menus.
          </>
        ),
      },
    ],
  },
} satisfies Meta<typeof Card>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {},
};
