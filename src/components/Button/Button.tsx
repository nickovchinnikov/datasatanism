import { FC, ReactNode, MouseEvent } from "react";

import styles from "./Button.module.scss";

type Variants = "primary" | "icon" | "context";

export interface Props {
  children: ReactNode;
  onClick?: (e: MouseEvent<HTMLButtonElement>) => void;
  variant?: "primary" | "icon" | "context";
  type?: "button" | "submit" | "reset";
  size?: "lg" | "xl";
}

const variantMap: Map<Variants, string> = new Map([
  ["primary", "buttonPrimary"],
  ["icon", "buttonIcon"],
  ["context", "buttonContent"],
]);

const contextButtons = new Set(["primary", "context"]);

export const Button: FC<Props> = ({
  children,
  onClick,
  type = "button",
  size = "lg",
  variant = "primary",
}) => {
  const buttonClass = styles[variantMap.get(variant) as string];
  const sizeClass =
    styles[`buttonSize${size.charAt(0).toUpperCase()}${size.slice(1)}`];

  return (
    <button
      className={`${styles.button} ${buttonClass} ${sizeClass}`}
      type={type}
      onClick={onClick}
    >
      {contextButtons.has(variant) && (
        <span className={styles.buttonContent}>{children}</span>
      )}
      {variant == "icon" && children}
    </button>
  );
};
