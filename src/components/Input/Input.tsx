import {
  FC,
  ChangeEvent,
  MouseEvent,
  useId,
  useState,
  useCallback,
} from "react";

import styles from "./Input.module.scss";

interface Props {
  label: string;
  placeholder?: string;
  defaultValue?: string | number;
  type?: "text" | "password" | "number";
  onChange?: (e: ChangeEvent<HTMLInputElement>) => void;
}

export const Input: FC<Props> = ({
  label,
  placeholder = "",
  defaultValue,
  type = "text",
  onChange = () => {},
}) => {
  const id = useId();
  const [value, setValue] = useState(defaultValue);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setValue(e.target.value);
      onChange(e);
    },
    [onChange],
  );

  const handleUp = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (!value) return;
      const newValue =
        typeof value === "number" ? value + 1 : parseInt(value) + 1;
      setValue(String(newValue));
    },
    [value],
  );

  const handleDown = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (!value) return;
      const newValue =
        typeof value === "number" ? value - 1 : parseInt(value) - 1;
      setValue(String(newValue));
    },
    [value],
  );

  return (
    <div className={styles.inputWrapper}>
      <label className={styles.inputLabel} htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        type={type}
        className={styles.input}
        placeholder={placeholder}
        value={value}
        onChange={handleChange}
      />
      {type == "number" && (
        <>
          <div
            className={`${styles.button} ${styles.buttonUp}`}
            onClick={handleUp}
          />
          <div
            className={`${styles.button} ${styles.buttonDown}`}
            onClick={handleDown}
          />
        </>
      )}
    </div>
  );
};
