import { FC, ReactNode, createElement } from "react";

import styles from "./Content.module.scss";

interface Props {
  children: ReactNode;
  className?: string;
}

export const ComponentCreator = ($component = "p") => {
  const TextComponent: FC<Props> = ({ children, className }) =>
    createElement(
      $component,
      {
        className: `${styles.textContent} ${className}`,
      },
      children,
    );
  return TextComponent;
};

export const Heading1 = ComponentCreator("h1");
export const Heading2 = ComponentCreator("h2");
export const Heading3 = ComponentCreator("h3");
export const Heading4 = ComponentCreator("h4");
export const Heading5 = ComponentCreator("h5");
export const Heading6 = ComponentCreator("h6");
export const Paragraph = ComponentCreator("p");
export const Span = ComponentCreator("span");
