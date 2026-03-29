package herdai

import (
	"encoding/json"
	"testing"
)

func TestOaiFlexContent_UnmarshalJSON(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"null", `null`, ""},
		{"empty string", `""`, ""},
		{"plain string", `"hello world"`, "hello world"},
		{"array of text parts", `[{"type":"text","text":"part1"},{"type":"text","text":"part2"}]`, "part1part2"},
		{"array with nested content key", `[{"content":"x"}]`, "x"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var c oaiFlexContent
			if err := json.Unmarshal([]byte(tc.raw), &c); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if c.Text != tc.want {
				t.Errorf("got %q want %q", c.Text, tc.want)
			}
		})
	}
}

func TestOaiResponseMessage_ArrayContent(t *testing.T) {
	raw := `{"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"Hello"}]}}]}`
	var resp oaiResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Choices) != 1 {
		t.Fatal("expected 1 choice")
	}
	if got := resp.Choices[0].Message.Content.Text; got != "Hello" {
		t.Errorf("content: got %q want Hello", got)
	}
}
