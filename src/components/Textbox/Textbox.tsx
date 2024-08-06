import React, {
  ChangeEvent,
  FC,
  useId,
  useState,
  useRef,
  useLayoutEffect,
  useCallback,
} from "react";

import stype from "./Textbox.module.scss";

interface Props {
  label: string;
  defaultValue?: string;
  onChange?: (e: ChangeEvent<HTMLTextAreaElement>) => void;
}

export const Textbox: FC<Props> = ({
  label,
  defaultValue,
  onChange = () => {},
}) => {
  const id = useId();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [value, setValue] = useState(defaultValue);

  function adjustHeight() {
    textareaRef!.current!.style.height = "inherit";
    textareaRef!.current!.style.height = `${textareaRef!.current!.scrollHeight}px`;
  }

  useLayoutEffect(adjustHeight, []);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      setValue(e.target.value);
      onChange(e);
    },
    [onChange],
  );

  return (
    <div className={stype.formGroup}>
      <label className={stype.formLabel} htmlFor={id}>
        {label}
      </label>
      <div className={stype.formControl}>
        <textarea
          id={id}
          onChange={handleChange}
          onKeyUp={adjustHeight}
          className={stype.formControl}
          name={label}
          ref={textareaRef}
          value={value}
        />
      </div>
    </div>
  );
};
