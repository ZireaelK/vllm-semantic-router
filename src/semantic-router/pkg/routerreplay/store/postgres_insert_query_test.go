package store

import (
	"regexp"
	"strings"
	"testing"
	"time"
)

// TestPostgresInsertQueryColumnArgsAlignment pins the three-way invariant
// between the INSERT column list, the $N placeholder list, and the slice
// returned by postgresInsertRecord.args().
//
// Background: PR #1857 added projection_trace to the column list but forgot
// to add $51 to the VALUES clause. Postgres rejected every write with
// "INSERT has more target columns than expressions"; the error was swallowed
// by persistReplayRecord and the dashboard Insight panel silently stayed
// empty for 25 days before anyone noticed.
func TestPostgresInsertQueryColumnArgsAlignment(t *testing.T) {
	cols := extractInsertColumns(t, postgresInsertQueryTemplate)
	if len(cols) == 0 {
		t.Fatalf("could not extract column list from postgresInsertQueryTemplate")
	}

	placeholders := extractInsertPlaceholders(t, postgresInsertQueryTemplate)
	if len(placeholders) == 0 {
		t.Fatalf("could not extract placeholder list from postgresInsertQueryTemplate")
	}

	args := samplePostgresInsertArgs(t)

	if len(cols) != len(placeholders) {
		t.Errorf("column / placeholder count mismatch: %d columns vs %d placeholders\ncolumns=%v\nplaceholders=%v",
			len(cols), len(placeholders), cols, placeholders)
	}
	if len(cols) != len(args) {
		t.Errorf("column / args count mismatch: %d columns vs %d args from postgresInsertRecord.args()",
			len(cols), len(args))
	}
	if len(placeholders) != len(args) {
		t.Errorf("placeholder / args count mismatch: %d placeholders vs %d args", len(placeholders), len(args))
	}

	for i, p := range placeholders {
		want := "$" + itoa(i+1)
		if p != want {
			t.Errorf("placeholder #%d is %q, expected %q (placeholders must be sequential $1..$N)", i+1, p, want)
		}
	}
}

// TestPostgresInsertSelectColumnAlignment pins the write path against the read
// path. postgresRecordRow.scanDestinations() is positional, so it only decodes
// correctly while postgresRecordSelectColumns lists the same columns in the same
// order as the INSERT statement. A column added to one list but not the other
// round trips as a zero value instead of failing loudly.
func TestPostgresInsertSelectColumnAlignment(t *testing.T) {
	insertCols := extractInsertColumns(t, postgresInsertQueryTemplate)
	selectCols := splitColumnList(postgresRecordSelectColumns)

	if len(insertCols) != len(selectCols) {
		t.Fatalf("insert / select column count mismatch: %d vs %d\ninsert=%v\nselect=%v",
			len(insertCols), len(selectCols), insertCols, selectCols)
	}
	for i := range insertCols {
		if insertCols[i] != selectCols[i] {
			t.Errorf("column #%d differs: insert=%q select=%q", i+1, insertCols[i], selectCols[i])
		}
	}

	row := postgresRecordRow{}
	if got := len(row.scanDestinations()); got != len(selectCols) {
		t.Errorf("scanDestinations() returns %d destinations for %d selected columns", got, len(selectCols))
	}
}

// extractInsertColumns parses the column list between
// `INSERT INTO %s (` and `) VALUES`.
func extractInsertColumns(t *testing.T, tmpl string) []string {
	t.Helper()
	start := strings.Index(tmpl, "INSERT INTO %s (")
	if start < 0 {
		return nil
	}
	start += len("INSERT INTO %s (")
	end := strings.Index(tmpl[start:], ") VALUES")
	if end < 0 {
		return nil
	}
	return splitColumnList(tmpl[start : start+end])
}

// splitColumnList turns a comma-separated SQL column list into trimmed names,
// dropping the whitespace and newlines both literals are formatted with.
func splitColumnList(raw string) []string {
	parts := strings.Split(raw, ",")
	cols := make([]string, 0, len(parts))
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if trimmed == "" {
			continue
		}
		cols = append(cols, trimmed)
	}
	return cols
}

// extractInsertPlaceholders parses the $N placeholders after `VALUES (`.
func extractInsertPlaceholders(t *testing.T, tmpl string) []string {
	t.Helper()
	re := regexp.MustCompile(`\$\d+`)
	valuesIdx := strings.Index(tmpl, "VALUES (")
	if valuesIdx < 0 {
		return nil
	}
	return re.FindAllString(tmpl[valuesIdx:], -1)
}

// samplePostgresInsertArgs builds a minimal Record, runs it through the
// production insert-record builder, and returns the resulting args slice.
// We do not assert specific values — only the slice length.
func samplePostgresInsertArgs(t *testing.T) []interface{} {
	t.Helper()
	rec := Record{
		ID:        "test-id",
		Timestamp: time.Now().UTC(),
	}
	built, err := newPostgresInsertRecord(rec)
	if err != nil {
		t.Fatalf("newPostgresInsertRecord: %v", err)
	}
	return built.args()
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// TestPostgresUpdateFusionQuorumQueryTemplate verifies that UpdateFusionQuorum
// generates an UPDATE query protected against both SQL NULL and JSONB 'null' literals.
//
// In PostgreSQL, uninitialized RouteDiagnostics records store JSONB literal 'null'::jsonb
// because typed nil pointers marshal to JSON "null". A naive COALESCE(route_diagnostics, '{}'::jsonb)
// evaluates 'null'::jsonb as non-null, causing jsonb_set to fail with "cannot call jsonb_set on a scalar".
//
// Normalizing via COALESCE(NULLIF(route_diagnostics, 'null'::jsonb), '{}'::jsonb) maps 'null'::jsonb
// to SQL NULL, which COALESCE safely converts to an empty JSONB object '{}'::jsonb before calling jsonb_set.
func TestPostgresUpdateFusionQuorumQueryTemplate(t *testing.T) {
	const tableName = "test_router_replay"
	query := postgresUpdateFusionQuorumQuery(tableName)

	if !strings.Contains(query, "UPDATE "+tableName) {
		t.Errorf("query does not target expected table %q: %s", tableName, query)
	}

	const expectedNullNormalization = "COALESCE(NULLIF(route_diagnostics, 'null'::jsonb), '{}'::jsonb)"
	if !strings.Contains(query, expectedNullNormalization) {
		t.Errorf("query does not normalize JSON 'null' to empty object;\nwant substring: %s\ngot query:\n%s",
			expectedNullNormalization, query)
	}

	// Verify the naive COALESCE without NULLIF is not present.
	naiveCoalesce := "COALESCE(route_diagnostics, '{}'::jsonb)"
	if strings.Contains(query, naiveCoalesce) {
		t.Errorf("query contains unsafe naive COALESCE without NULLIF guard: %s", query)
	}

	for _, expectedClause := range []string{
		"route_diagnostics = jsonb_set(",
		"'{fusion_quorum}'",
		"$2::jsonb",
		"true",
		"WHERE id = $1",
	} {
		if !strings.Contains(query, expectedClause) {
			t.Errorf("query missing expected clause %q:\n%s", expectedClause, query)
		}
	}
}

// TestPostgresRouteDiagnosticsTypedNilMarshalingContract pins the encoding contract
// between Go typed nil pointers (*RouteDiagnostics), JSON marshaling, and PostgreSQL JSONB storage:
//
//  1. A Record with RouteDiagnostics == nil has a typed nil pointer (*RouteDiagnostics)(nil).
//  2. Passing it to marshalReplayOptionalJSON(interface{}) yields interface{}((*RouteDiagnostics)(nil)).
//     Because a typed nil interface is not nil, json.Marshal encodes it to []byte("null").
//  3. PostgreSQL JSONB columns persist []byte("null") as JSON literal 'null'::jsonb (scalar, non-null in SQL).
//  4. unmarshalReplayOptionalJSON decodes []byte("null") back to a nil pointer (preserving contract).
//  5. When RouteDiagnostics is populated, it encodes to a JSON object.
func TestPostgresRouteDiagnosticsTypedNilMarshalingContract(t *testing.T) {
	// Case 1: uninitialized record with typed nil pointer (*RouteDiagnostics)(nil)
	emptyRec := Record{
		ID:        "test-uninit-route-diag",
		Timestamp: time.Now().UTC(),
	}
	if emptyRec.RouteDiagnostics != nil {
		t.Fatal("expected RouteDiagnostics to be nil initially")
	}

	insert, err := newPostgresInsertRecord(emptyRec)
	if err != nil {
		t.Fatalf("newPostgresInsertRecord: %v", err)
	}

	idx := columnIndex(t, "route_diagnostics")
	args := insert.args()
	if idx >= len(args) {
		t.Fatalf("column index %d out of bounds for args (len %d)", idx, len(args))
	}

	diagArg, ok := args[idx].([]byte)
	if !ok {
		t.Fatalf("route_diagnostics arg is not []byte: %T (%v)", args[idx], args[idx])
	}

	// Typed nil pointer marshals to JSON literal "null"
	if string(diagArg) != "null" {
		t.Fatalf("expected route_diagnostics arg to be []byte(\"null\") for typed nil, got %q", string(diagArg))
	}

	// Verify decoding: []byte("null") unmarshals back to nil RouteDiagnostics pointer
	var decoded *RouteDiagnostics
	if err := unmarshalReplayOptionalJSON(diagArg, &decoded); err != nil {
		t.Fatalf("unmarshalReplayOptionalJSON failed: %v", err)
	}
	if decoded != nil {
		t.Fatalf("expected unmarshaled RouteDiagnostics to be nil, got %+v", decoded)
	}

	// Case 2: populated record with non-nil RouteDiagnostics
	populatedRec := Record{
		ID:        "test-populated-route-diag",
		Timestamp: time.Now().UTC(),
		RouteDiagnostics: &RouteDiagnostics{
			SelectionMethod: "fusion",
		},
	}
	insertPopulated, err := newPostgresInsertRecord(populatedRec)
	if err != nil {
		t.Fatalf("newPostgresInsertRecord: %v", err)
	}
	populatedArg, ok := insertPopulated.args()[idx].([]byte)
	if !ok {
		t.Fatalf("populated route_diagnostics arg is not []byte: %T", insertPopulated.args()[idx])
	}
	if !strings.Contains(string(populatedArg), `"selection_method":"fusion"`) {
		t.Fatalf("expected serialized JSON to contain selection_method, got %s", string(populatedArg))
	}
}
