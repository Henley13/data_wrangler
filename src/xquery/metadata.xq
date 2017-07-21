declare namespace functx = "http://www.functx.com";
declare function functx:wrap-values-in-elements
  ( $values as xs:anyAtomicType* ,
    $elementName as xs:QName )  as element()* {

   for $value in $values
   return element {$elementName} {$value}
 } ;
 
<results>
{
for $page in collection('metadata')//metadata
    let $id := $page/id
    let $title := $page/title
    let $a := $page/organization/id
    let $b := $page/organization/title
    let $idorg := functx:wrap-values-in-elements($a, xs:QName('idorg'))
    let $titleorg := functx:wrap-values-in-elements($b, xs:QName('titleorg'))

    for $table in $page/tables/table
        where $page/tables/table/url_destination = 'file'
        let $c := $table/id
        let $idtable := functx:wrap-values-in-elements($c, xs:QName('idtable'))
        let $url := $table/url

        return <page>{$id, $title, $idtable, $idorg, $titleorg, $url}</page>
}
</results>
