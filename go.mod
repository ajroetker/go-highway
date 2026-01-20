module github.com/ajroetker/go-highway

go 1.26

require golang.org/x/sys v0.40.0

require (
	github.com/gorse-io/goat v0.1.3 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/klauspost/asmfmt v1.3.2 // indirect
	github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec // indirect
	github.com/samber/lo v1.50.0 // indirect
	github.com/spf13/cobra v1.7.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/text v0.22.0 // indirect
	modernc.org/cc/v4 v4.26.3 // indirect
	modernc.org/mathutil v1.7.1 // indirect
	modernc.org/opt v0.1.4 // indirect
	modernc.org/sortutil v1.2.1 // indirect
	modernc.org/strutil v1.2.1 // indirect
	modernc.org/token v1.1.0 // indirect
)

// Use ajroetker/goat fork with SVE/SME streaming mode, FP16 headers, ABI offset fixes, int32_t support, size-appropriate load instructions, and stack frame fixes
replace github.com/gorse-io/goat => github.com/ajroetker/goat v0.0.0-sve-sme-support-013

tool github.com/gorse-io/goat
