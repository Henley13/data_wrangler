<results>
{
for $table in collection('metadata')//tables/table
let $id := $table/id
let $url := $table/url
let $url_destination := $table/url_destination
let $format := $table/format
where $url_destination = 'file'
where $format = ('xml', 'XML')
return <table>{$url, $id, $format}</table>
}
</results>