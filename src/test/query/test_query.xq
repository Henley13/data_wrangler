<results>
{
for $table in collection('test_database')//tables/table
let $id := $table/id
let $url := $table/url
let $url_destination := $table/url_destination
let $format := $table/format
where $url_destination = 'file'
return <table>{$url, $id, $format}</table>
}
</results>