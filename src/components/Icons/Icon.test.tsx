import { render } from "@testing-library/react";

import { Icon } from "./Icon";

describe("Icon test cases", () => {
  it("Icon render check", () => {
    const { asFragment } = render(<Icon name="Moon" />);

    expect(asFragment()).toMatchSnapshot();
  });
});
