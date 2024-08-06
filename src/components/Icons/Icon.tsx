import { FC } from "react";

import * as Icons from "./Icons";

export type AvailableIcons = keyof typeof Icons;

export type Props = {
  /** Icon name */
  name: AvailableIcons;
  /** Width and height */
  size?: number;
} & React.SVGProps<SVGSVGElement>;

// https://reactsvgicons.com/search

export const SizeInRem = (size: number) => {
  const sizeInRem = `${size}rem`;
  return sizeInRem;
};

export const Icon: FC<Props> = ({ name, size = 2, ...rest }) => {
  const Icon = Icons[name];
  const sizeInRem = SizeInRem(size);
  const sizes = { width: sizeInRem, height: sizeInRem };

  return <Icon role="img" aria-label={name} {...sizes} {...rest} />;
};
