<results>
{
for $table in collection('metadata')//tables/table
let $id := $table/id
let $url := $table/url
let $url_destination := $table/url_destination
where $url_destination = 'file'
return <table>{$url, $id}</table>
}
</results>
